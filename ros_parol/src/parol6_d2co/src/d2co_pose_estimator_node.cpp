// d2co_pose_estimator_node.cpp — Nodo ROS 2 de estimación de pose 6D del tubo
// mediante D2CO (Direct Directional Chamfer Optimization, Imperoli & Pretto,
// ICVS 2015), integrando la librería externa libd2co en el pipeline de visión
// de PAROL6.
//
// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PIPELINE                                                                 ║
// ║                                                                           ║
// ║  1. YOLO detecta el tubo en 2D          → vision_msgs/Detection2DArray    ║
// ║  2. El bbox 2D define una máscara (ROI) → solo bordes dentro del bbox     ║
// ║  3. Canny/LSD + Distance Transform 3D   → tensor direccional (60 canales) ║
// ║  4. Hipótesis iniciales desde el bbox   → rayo + z estimada + eje tubo    ║
// ║  5. D2CO refina con Levenberg-Marquardt → pose 6D (modelo→cámara)         ║
// ║  6. Se publica geometry_msgs/PoseStamped (mismo contrato que el nodo ICP) ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
// ENTRADAS
//   /vision/detections                    vision_msgs/Detection2DArray (YOLO)
//   <image_topic>                         sensor_msgs/Image  (mono o color)
//   <camera_info_topic>                   sensor_msgs/CameraInfo
//
// SALIDAS
//   /vision/tube_pose                     geometry_msgs/PoseStamped (6D)
//   /pose_target_cuaterniones             geometry_msgs/Pose (opcional, legacy)
//   /vision/d2co_overlay                  sensor_msgs/Image (debug RViz)
//   TF: <output_frame> -> tube_link_estimated
//
// SERVICIO
//   ~/estimate_now  std_srvs/Trigger — fuerza una estimación inmediata.
//
// ─────────────────────────────────────────────────────────────────────────────
// GESTIÓN DEL ROI (importante)
//   D2CO soporta ROI nativo vía cv_ext::PinholeCameraModel::setRegionOfInterest,
//   pero RasterObjectModel3D crea su ventana OpenGL (GLFW) UNA sola vez en
//   computeRaster() y el ROI queda congelado en ese instante. Como el bbox de
//   YOLO cambia en cada frame, aquí NO se usa el ROI nativo; en su lugar:
//     • La imagen se procesa COMPLETA pero reescalada (image_scale, def. 0.5),
//       con la intrínseca escalada de forma consistente por
//       PinholeCameraModel::setSizeScaleFactor (no hay que tocar K a mano).
//     • El bbox se aplica como MÁSCARA del detector de bordes
//       (EdgeDetector::setMask): el tensor de distancias solo contiene bordes
//       del entorno del tubo → mismo efecto que el recorte, sin reinicializar
//       el contexto OpenGL y sin bordes espurios en la frontera del recorte.
// ─────────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// ── Librería D2CO (externa, no-ROS) ─────────────────────────────────────────
#include "raster_object_model3D.h"
#include "chamfer_matching.h"
#include "distance_transforms.h"
#include "edge_detectors.h"

using namespace std::chrono_literals;

namespace
{

/// Hipótesis de pose inicial (modelo→cámara) para el optimizador D2CO.
struct PoseHypothesis
{
  Eigen::Quaterniond q;
  Eigen::Vector3d t;
  double score = std::numeric_limits<double>::max();
};

/// Estadísticas de los bordes dentro de la máscara del bbox (en px de la
/// imagen de trabajo). Los bbox de YOLO pueden ser holgados Y descentrados,
/// así que el centro/ángulo/longitud del objeto se estiman de los propios
/// bordes (centroide + PCA), no de la caja.
struct EdgeStats
{
  bool valid = false;
  double u = 0.0, v = 0.0;    // centroide de los bordes
  double theta = 0.0;         // ángulo del eje mayor (PCA) en la imagen
  bool theta_valid = false;   // false si los bordes no son alargados
  double len_px = 0.0;        // extensión a lo largo del eje mayor
  double wid_px = 0.0;        // separación de raíles (eje menor) — clave:
                              // mide la profundidad SIN ambigüedad de
                              // inclinación (z = fx·Ø/wid)
};

EdgeStats computeEdgeStats(const cv::Mat & edges)
{
  EdgeStats st;

  // La máscara del bbox puede contener bordes de clutter (relieve del
  // terreno) además del objeto: clusters contiguos que arruinan media y
  // varianza. Robustez: dilatar para unir los raíles del contorno del tubo
  // y quedarse SOLO con la componente conexa de mayor área.
  cv::Mat dilated;
  cv::dilate(edges, dilated, cv::getStructuringElement(cv::MORPH_RECT,
                                                       cv::Size(3, 3)));
  cv::Mat labels, stats, centroids;
  const int n = cv::connectedComponentsWithStats(dilated, labels, stats,
                                                 centroids, 8);
  if (n < 2) {
    return st;
  }
  int best_label = -1, best_area = 0;
  for (int i = 1; i < n; ++i) {
    const int area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (area > best_area) {
      best_area = area;
      best_label = i;
    }
  }

  std::vector<cv::Point> all_pts;
  cv::findNonZero(edges, all_pts);
  std::vector<cv::Point> pts;
  pts.reserve(all_pts.size());
  for (const auto & p : all_pts) {
    if (labels.at<int>(p) == best_label) {
      pts.push_back(p);
    }
  }
  if (pts.size() < 20) {
    return st;
  }

  cv::Mat data(static_cast<int>(pts.size()), 2, CV_64FC1);
  for (size_t i = 0; i < pts.size(); ++i) {
    data.at<double>(static_cast<int>(i), 0) = pts[i].x;
    data.at<double>(static_cast<int>(i), 1) = pts[i].y;
  }
  cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
  st.u = pca.mean.at<double>(0, 0);
  st.v = pca.mean.at<double>(0, 1);
  const double ex = pca.eigenvectors.at<double>(0, 0);
  const double ey = pca.eigenvectors.at<double>(0, 1);
  st.theta = std::atan2(ey, ex);
  const double l0 = pca.eigenvalues.at<double>(0, 0);
  const double l1 = std::max(pca.eigenvalues.at<double>(1, 0), 1e-9);
  // Solo fiarse del ángulo si la nube de bordes es claramente alargada.
  st.theta_valid = (l0 / l1) > 4.0;

  // Extensión a lo largo del eje mayor: rango percentil p2..p98 de las
  // proyecciones (robusto frente a píxeles sueltos dentro del componente).
  std::vector<double> proj(pts.size()), perp(pts.size());
  for (size_t i = 0; i < pts.size(); ++i) {
    proj[i] = (pts[i].x - st.u) * ex + (pts[i].y - st.v) * ey;
    perp[i] = -(pts[i].x - st.u) * ey + (pts[i].y - st.v) * ex;
  }
  std::sort(proj.begin(), proj.end());
  std::sort(perp.begin(), perp.end());
  const size_t i_lo = static_cast<size_t>(0.02 * (proj.size() - 1));
  const size_t i_hi = static_cast<size_t>(0.98 * (proj.size() - 1));
  st.len_px = proj[i_hi] - proj[i_lo];
  const size_t j_lo = static_cast<size_t>(0.05 * (perp.size() - 1));
  const size_t j_hi = static_cast<size_t>(0.95 * (perp.size() - 1));
  st.wid_px = perp[j_hi] - perp[j_lo];

  st.valid = true;
  return st;
}

}  // namespace

class D2coPoseEstimatorNode : public rclcpp::Node
{
public:
  D2coPoseEstimatorNode()
  : Node("d2co_pose_estimator_node")
  {
    // ═══════════════ PARÁMETROS ════════════════════════════════════════════
    // ── Tópicos ─────────────────────────────────────────────────────────────
    declare_parameter("detections_topic", "/vision/detections");
    declare_parameter("image_topic", "/navcam/left/navcam_left/image_raw");
    declare_parameter("camera_info_topic",
                      "/navcam/left/navcam_left/camera_info");
    declare_parameter("pose_topic", "/vision/tube_pose");
    declare_parameter("legacy_pose_topic", "/pose_target_cuaterniones");
    declare_parameter("overlay_debug_topic", "/vision/d2co_overlay");

    // ── Salida / frames ─────────────────────────────────────────────────────
    declare_parameter("output_frame", "");   // "" => frame óptico de la cámara
    declare_parameter("tf_child_frame", "tube_link_estimated");
    declare_parameter("broadcast_tf", true);
    declare_parameter("publish_legacy_pose", false);
    declare_parameter("publish_debug_overlay", true);

    // ── Modelo CAD ──────────────────────────────────────────────────────────
    // Vacío => share/parol6_d2co/models/mars2020tube_scaled.stl
    // ¡Debe ser el STL PRE-ESCALADO (en metros)! D2CO solo aplica escala
    // uniforme (unidad de medida), no la escala anisótropa del STL original.
    declare_parameter("stl_path", "");
    // Paso de muestreo del raster del modelo. A 512² de trabajo, 1 px ≈
    // 1.7 mm a z=0.7 m: muestrear más fino que ~2 mm solo añade residuales
    // redundantes al optimizador (coste lineal en puntos).
    declare_parameter("step_meters", 0.002);

    // ── D2CO ────────────────────────────────────────────────────────────────
    declare_parameter("image_scale", 0.5);     // escala de la imagen procesada
    declare_parameter("num_directions", 60);   // canales del tensor (paper: 60)
    declare_parameter("tensor_lambda", 6.0);   // peso direccional λ (paper: 6)
    declare_parameter("optimizer_iterations", 50);  // iteraciones LM (Ceres)
    declare_parameter("edge_detector", "CANNY");    // CANNY | LSD
    declare_parameter("canny_low_threshold", 40);
    declare_parameter("canny_ratio", 2);
    // Saturación de la transformada de distancias (px). Acota la influencia
    // de los puntos del modelo lejos de todo borde y hace el ranking de
    // hipótesis mucho más discriminante en escenas con pocos bordes.
    declare_parameter("dist_threshold", 30.0);

    // ── Detección YOLO / ROI ────────────────────────────────────────────────
    declare_parameter("min_score", 0.3);
    declare_parameter("target_class", "");
    declare_parameter("bbox_margin_px", 24.0);    // margen de la máscara (px
                                                  // de la imagen original)
    // El YOLO de serie (skip_frames=10 a 5 Hz) publica cada ~2 s.
    declare_parameter("detection_max_age_s", 2.5);

    // ── Hipótesis iniciales ─────────────────────────────────────────────────
    declare_parameter("tube_length", 0.186);    // m (mars2020tube escalado)
    declare_parameter("tube_diameter", 0.0208); // m — ancla la profundidad
                                                // sin ambigüedad de tilt
    declare_parameter("width_bias_px", 1.0);    // sesgo del ancho Canny (px)
    // Solo se corrige por anchura si el ajuste D2CO es pobre (avg > gate):
    // con buen ajuste, el optimizador es más preciso que la medida de ancho.
    declare_parameter("width_correction_gate", 1.5);
    declare_parameter("z_min", 0.3);           // m, profundidad mínima
    declare_parameter("z_max", 2.0);           // m, profundidad máxima
    // Barrido de profundidad MULTIPLICATIVO sobre la z estimada del bbox.
    // Los bbox de YOLO suelen ser holgados => la z por tamaño aparente sale
    // SUBESTIMADA; por eso el barrido se sesga hacia factores > 1.
    declare_parameter("z_factors",
                      std::vector<double>{0.8, 1.0, 1.35, 1.8});
    declare_parameter("tilt_angles_deg",
                      std::vector<double>{-40.0, -20.0, 0.0, 20.0, 40.0});
    declare_parameter("refine_top_k", 3);      // nº de hipótesis a refinar
    // Umbral de aceptación sobre la distancia media en el tensor. Con la DT
    // saturada a 30 px, los rasgos AUSENTES (p.ej. tapas del tubo sin
    // contraste) añaden ~3 puntos sin implicar desalineación; los priors
    // geométricos de abajo son los que garantizan la pose.
    declare_parameter("max_avg_dist", 4.5);
    // Priors geométricos para desempatar el valle plano de profundidad de un
    // cilindro monocular: consistencia de z con la longitud medida de los
    // bordes (z* = fx·L·cosφ/l_px) y del centro proyectado con el centroide.
    declare_parameter("depth_consistency_weight", 5.0);
    declare_parameter("center_consistency_weight", 0.05);  // por px
    declare_parameter("tracking_max_age_s", 3.0);
    // Atajo de tracking: si la mejor semilla ya puntúa por debajo de este
    // umbral (típicamente la pose del ciclo anterior), se refina solo esa.
    // Calibrado en simulación: los ajustes correctos puntúan ≤0.6 y los
    // mínimos locales de profundidad errónea ≥0.9.
    declare_parameter("tracking_fast_dist", 0.6);
    // Cada N ciclos se ignora el atajo y se refinan refine_top_k hipótesis,
    // para poder escapar de mínimos locales fijados por el tracking.
    declare_parameter("full_search_every", 5);

    declare_parameter("process_rate", 1.0);    // Hz

    // ── Lectura ─────────────────────────────────────────────────────────────
    image_topic_       = get_parameter("image_topic").as_string();
    output_frame_      = get_parameter("output_frame").as_string();
    tf_child_frame_    = get_parameter("tf_child_frame").as_string();
    broadcast_tf_      = get_parameter("broadcast_tf").as_bool();
    pub_legacy_        = get_parameter("publish_legacy_pose").as_bool();
    pub_overlay_       = get_parameter("publish_debug_overlay").as_bool();
    stl_path_          = get_parameter("stl_path").as_string();
    step_meters_       = get_parameter("step_meters").as_double();
    image_scale_       = get_parameter("image_scale").as_double();
    num_directions_    = static_cast<int>(get_parameter("num_directions").as_int());
    tensor_lambda_     = get_parameter("tensor_lambda").as_double();
    optimizer_iters_   = static_cast<int>(get_parameter("optimizer_iterations").as_int());
    edge_detector_type_ = get_parameter("edge_detector").as_string();
    canny_low_thresh_  = static_cast<int>(get_parameter("canny_low_threshold").as_int());
    canny_ratio_       = static_cast<int>(get_parameter("canny_ratio").as_int());
    dist_threshold_    = get_parameter("dist_threshold").as_double();
    min_score_         = get_parameter("min_score").as_double();
    target_class_      = get_parameter("target_class").as_string();
    bbox_margin_px_    = get_parameter("bbox_margin_px").as_double();
    det_max_age_s_     = get_parameter("detection_max_age_s").as_double();
    tube_length_       = get_parameter("tube_length").as_double();
    tube_diameter_     = get_parameter("tube_diameter").as_double();
    width_bias_px_     = get_parameter("width_bias_px").as_double();
    width_correction_gate_ = get_parameter("width_correction_gate").as_double();
    z_min_             = get_parameter("z_min").as_double();
    z_max_             = get_parameter("z_max").as_double();
    z_factors_         = get_parameter("z_factors").as_double_array();
    tilt_angles_deg_   = get_parameter("tilt_angles_deg").as_double_array();
    refine_top_k_      = static_cast<int>(get_parameter("refine_top_k").as_int());
    max_avg_dist_      = get_parameter("max_avg_dist").as_double();
    w_depth_           = get_parameter("depth_consistency_weight").as_double();
    w_center_          = get_parameter("center_consistency_weight").as_double();
    tracking_max_age_s_ = get_parameter("tracking_max_age_s").as_double();
    tracking_fast_dist_ = get_parameter("tracking_fast_dist").as_double();
    full_search_every_ = static_cast<int>(get_parameter("full_search_every").as_int());
    const double process_rate = get_parameter("process_rate").as_double();

    std::transform(edge_detector_type_.begin(), edge_detector_type_.end(),
                   edge_detector_type_.begin(), ::toupper);

    // ── Ruta por defecto del STL escalado ───────────────────────────────────
    if (stl_path_.empty()) {
      stl_path_ = ament_index_cpp::get_package_share_directory("parol6_d2co")
                  + "/models/mars2020tube_scaled.stl";
    }

    // RasterObjectModel3D necesita un contexto OpenGL (GLFW): sin servidor
    // gráfico el nodo no puede funcionar. Mejor avisar pronto y claro.
    if (!std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY")) {
      RCLCPP_FATAL(get_logger(),
                   "D2CO necesita un contexto OpenGL (GLFW) y no hay DISPLAY/"
                   "WAYLAND_DISPLAY. Ejecuta en una sesion grafica o con xvfb.");
      throw std::runtime_error("Sin display para el contexto OpenGL de D2CO");
    }

    // ═══════════════ SUSCRIPTORES ═══════════════════════════════════════════
    // QoS BEST_EFFORT para los sensores de Gazebo (igual que el nodo ICP).
    auto sensor_qos = rclcpp::SensorDataQoS().keep_last(5);

    sub_detections_ = create_subscription<vision_msgs::msg::Detection2DArray>(
      get_parameter("detections_topic").as_string(), 10,
      std::bind(&D2coPoseEstimatorNode::detectionsCb, this,
                std::placeholders::_1));
    sub_camera_info_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      get_parameter("camera_info_topic").as_string(), sensor_qos,
      std::bind(&D2coPoseEstimatorNode::cameraInfoCb, this,
                std::placeholders::_1));
    sub_image_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_, sensor_qos,
      std::bind(&D2coPoseEstimatorNode::imageCb, this, std::placeholders::_1));

    // ═══════════════ PUBLICADORES ═══════════════════════════════════════════
    pub_pose_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      get_parameter("pose_topic").as_string(), 10);
    pub_legacy_pose_ = create_publisher<geometry_msgs::msg::Pose>(
      get_parameter("legacy_pose_topic").as_string(), 10);
    pub_overlay_image_ = create_publisher<sensor_msgs::msg::Image>(
      get_parameter("overlay_debug_topic").as_string(), 1);

    // ═══════════════ TF ═════════════════════════════════════════════════════
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // ═══════════════ SERVICIO + TIMER ═══════════════════════════════════════
    srv_estimate_ = create_service<std_srvs::srv::Trigger>(
      "~/estimate_now",
      std::bind(&D2coPoseEstimatorNode::estimateNowCb, this,
                std::placeholders::_1, std::placeholders::_2));

    const double period = 1.0 / std::max(process_rate, 0.1);
    timer_ = create_wall_timer(
      std::chrono::duration<double>(period),
      std::bind(&D2coPoseEstimatorNode::timerCb, this));

    RCLCPP_INFO(get_logger(),
                "D2coPoseEstimatorNode iniciado. Modelo: %s | escala imagen: "
                "%.2f | %d direcciones | lambda=%.1f | procesando a %.1f Hz",
                stl_path_.c_str(), image_scale_, num_directions_,
                tensor_lambda_, process_rate);
  }

private:
  // ══════════════════════════════════════════════════════════════════════════
  // CALLBACKS DE SUSCRIPCIÓN (solo cachean; el procesado lo dispara el timer)
  // ══════════════════════════════════════════════════════════════════════════

  void cameraInfoCb(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!have_camera_info_) {
      RCLCPP_INFO(get_logger(),
                  "Intrinseca recibida: fx=%.2f fy=%.2f cx=%.2f cy=%.2f "
                  "(%ux%u)",
                  msg->k[0], msg->k[4], msg->k[2], msg->k[5],
                  msg->width, msg->height);
    }
    camera_info_ = *msg;
    have_camera_info_ = true;
  }

  void imageCb(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(*msg, "mono8");
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_WARN(get_logger(), "CvBridge (image): %s", e.what());
      return;
    }
    std::lock_guard<std::mutex> lk(mutex_);
    last_image_ = cv_ptr->image;
    last_image_stamp_ = msg->header.stamp;
    image_frame_ = msg->header.frame_id;
  }

  void detectionsCb(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
  {
    // Selecciona la mejor detección (misma lógica que el nodo ICP).
    double best_score = -1.0;
    cv::Rect2d best_bbox;
    for (const auto & det : msg->detections) {
      if (det.results.empty()) {
        continue;
      }
      const auto & hyp = det.results[0].hypothesis;
      if (!target_class_.empty() && hyp.class_id != target_class_) {
        continue;
      }
      if (hyp.score < min_score_ || hyp.score <= best_score) {
        continue;
      }
      best_score = hyp.score;
      const double cx = det.bbox.center.position.x;
      const double cy = det.bbox.center.position.y;
      best_bbox = cv::Rect2d(cx - det.bbox.size_x / 2.0,
                             cy - det.bbox.size_y / 2.0,
                             det.bbox.size_x, det.bbox.size_y);
    }
    if (best_score < 0.0) {
      return;
    }
    std::lock_guard<std::mutex> lk(mutex_);
    last_bbox_ = best_bbox;
    last_bbox_score_ = best_score;
    last_detection_time_ = now();
  }

  // ══════════════════════════════════════════════════════════════════════════
  // DISPARADORES
  // ══════════════════════════════════════════════════════════════════════════

  void timerCb() { runPipeline("[timer]"); }

  void estimateNowCb(const std_srvs::srv::Trigger::Request::SharedPtr,
                     std_srvs::srv::Trigger::Response::SharedPtr res)
  {
    auto [ok, info] = runPipeline("[service]");
    res->success = ok;
    res->message = info;
  }

  // ══════════════════════════════════════════════════════════════════════════
  // INICIALIZACIÓN PEREZOSA DE D2CO (necesita la intrínseca y el tamaño real)
  // ══════════════════════════════════════════════════════════════════════════

  bool initD2co()
  {
    if (d2co_ready_) {
      return true;
    }

    sensor_msgs::msg::CameraInfo info;
    {
      std::lock_guard<std::mutex> lk(mutex_);
      if (!have_camera_info_) {
        return false;
      }
      info = camera_info_;
    }

    // ── Modelo pinhole de cv_ext a partir del CameraInfo ────────────────────
    cv::Mat K = (cv::Mat_<double>(3, 3)
                 << info.k[0], info.k[1], info.k[2],
                    info.k[3], info.k[4], info.k[5],
                    info.k[6], info.k[7], info.k[8]);
    cv::Mat dist;
    bool has_dist = false;
    for (const double d : info.d) {
      if (std::abs(d) > 1e-12) {
        has_dist = true;
      }
    }
    if (has_dist) {
      dist = cv::Mat(static_cast<int>(info.d.size()), 1, CV_64FC1);
      for (size_t i = 0; i < info.d.size(); ++i) {
        dist.at<double>(static_cast<int>(i), 0) = info.d[i];
      }
    }
    cam_model_ = cv_ext::PinholeCameraModel(
      K, static_cast<int>(info.width), static_cast<int>(info.height), dist);
    // El reescalado uniforme de la imagen lo gestiona el propio modelo de
    // cámara: fx,fy,cx,cy se escalan de forma consistente (no tocar K a mano).
    // OJO: el argumento de setSizeScaleFactor es un DIVISOR (factor 2 =>
    // mitad de resolución), por eso se pasa el inverso de image_scale.
    cam_model_.setSizeScaleFactor(1.0 / image_scale_);
    // Escala de trabajo EFECTIVA (px trabajo / px originales), derivada del
    // propio modelo para ser inmune a redondeos.
    work_scale_ = static_cast<double>(cam_model_.imgWidth()) /
                  static_cast<double>(info.width);

    // ── Modelo raster 3D del tubo (STL PRE-escalado, en metros) ─────────────
    obj_model_ = std::make_shared<RasterObjectModel3D>();
    obj_model_->setCamModel(cam_model_);
    obj_model_->setStepMeters(step_meters_);
    // Origen en el centro del bounding box ⇒ la pose 6D es la del CENTRO
    // geométrico del tubo (misma convención que el nodo ICP / tube_link).
    obj_model_->setBBCenterOrigOffset();
    if (!obj_model_->setModelFile(stl_path_)) {
      RCLCPP_ERROR(get_logger(), "No se pudo cargar el modelo: %s",
                   stl_path_.c_str());
      obj_model_.reset();
      return false;
    }
    // computeRaster crea la ventana OpenGL oculta UNA única vez: a partir de
    // aquí el modelo de cámara del raster queda congelado (ver cabecera).
    obj_model_->computeRaster();

    // ── Optimizador D2CO ────────────────────────────────────────────────────
    matcher_ = std::make_shared<DirectionalChamferMatching>();
    matcher_->setNumDirections(num_directions_);
    matcher_->setTemplateModel(obj_model_);
    matcher_->setOptimizerNumIterations(optimizer_iters_);
    matcher_->enableVerbouseMode(false);

    d2co_ready_ = true;
    RCLCPP_INFO(get_logger(),
                "D2CO listo: raster del modelo con paso %.1f mm, imagen de "
                "trabajo %dx%d.",
                step_meters_ * 1000.0, cam_model_.imgWidth(),
                cam_model_.imgHeight());
    return true;
  }

  // ══════════════════════════════════════════════════════════════════════════
  // PIPELINE
  // ══════════════════════════════════════════════════════════════════════════

  std::pair<bool, std::string> runPipeline(const char * log_prefix)
  {
    if (busy_) {
      return {false, "ocupado: estimacion anterior en curso"};
    }
    busy_ = true;
    std::pair<bool, std::string> result;
    try {
      result = pipelineImpl(log_prefix);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "%s fallo inesperado: %s", log_prefix,
                   e.what());
      result = {false, std::string("excepcion: ") + e.what()};
    }
    busy_ = false;
    return result;
  }

  std::pair<bool, std::string> pipelineImpl(const char * log_prefix)
  {
    if (!initD2co()) {
      return {false, "sin camera_info todavia"};
    }

    // ── 0. Copia coherente del estado cacheado ──────────────────────────────
    cv::Mat image;
    cv::Rect2d bbox;
    double bbox_score = 0.0;
    rclcpp::Time det_time;
    builtin_interfaces::msg::Time img_stamp;
    std::string frame;
    {
      std::lock_guard<std::mutex> lk(mutex_);
      if (last_image_.empty()) {
        return {false, "sin imagen todavia"};
      }
      if (last_bbox_score_ < 0.0) {
        return {false, "sin deteccion YOLO todavia"};
      }
      image = last_image_.clone();
      img_stamp = last_image_stamp_;
      frame = image_frame_;
      bbox = last_bbox_;
      bbox_score = last_bbox_score_;
      det_time = last_detection_time_;
    }

    // Detecciones obsoletas (YOLO dejó de publicar) no deben generar poses.
    const double det_age = (now() - det_time).seconds();
    if (det_age > det_max_age_s_) {
      return {false, "deteccion YOLO obsoleta (" + std::to_string(det_age) +
                     " s)"};
    }

    // ── 1. Imagen de trabajo reescalada + máscara ROI del bbox ──────────────
    const int w = cam_model_.imgWidth(), h = cam_model_.imgHeight();
    cv::Mat work_img;
    cv::resize(image, work_img, cv::Size(w, h), 0.0, 0.0, cv::INTER_AREA);

    const double s = work_scale_;
    cv::Rect roi(
      static_cast<int>(std::floor((bbox.x - bbox_margin_px_) * s)),
      static_cast<int>(std::floor((bbox.y - bbox_margin_px_) * s)),
      static_cast<int>(std::ceil((bbox.width + 2.0 * bbox_margin_px_) * s)),
      static_cast<int>(std::ceil((bbox.height + 2.0 * bbox_margin_px_) * s)));
    roi &= cv::Rect(0, 0, w, h);
    if (roi.width < 8 || roi.height < 8) {
      return {false, "ROI degenerado tras reescalar el bbox"};
    }
    cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);
    mask(roi).setTo(255);

    // ── 1b. Estadísticas de los bordes del objeto (centroide + PCA) ─────────
    // Los bbox de YOLO pueden venir holgados y DESCENTRADOS: las semillas se
    // anclan al centroide/eje real de los bordes, no al centro de la caja.
    // Se miden a RESOLUCIÓN COMPLETA (el ruido de cuantización de ±0.5 px
    // importa: la anchura de raíles ancla la profundidad) y se convierten a
    // coordenadas de trabajo conservando el subpíxel.
    cv::Rect roi_full(
      static_cast<int>(std::floor(bbox.x - bbox_margin_px_)),
      static_cast<int>(std::floor(bbox.y - bbox_margin_px_)),
      static_cast<int>(std::ceil(bbox.width + 2.0 * bbox_margin_px_)),
      static_cast<int>(std::ceil(bbox.height + 2.0 * bbox_margin_px_)));
    roi_full &= cv::Rect(0, 0, image.cols, image.rows);
    cv::Mat full_mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    full_mask(roi_full).setTo(255);
    cv::Mat seed_edges;
    cv::Canny(image, seed_edges, canny_low_thresh_,
              canny_low_thresh_ * canny_ratio_);
    seed_edges &= full_mask;
    EdgeStats edge_stats = computeEdgeStats(seed_edges);
    if (edge_stats.valid) {
      edge_stats.u *= s;
      edge_stats.v *= s;
      edge_stats.len_px *= s;
      // La ANCHURA se mide sobre los bordes a resolución de trabajo: a
      // resolución completa los puntos de las tapas pueblan el interior y
      // estrechan los percentiles perpendiculares (calibrado en simulación).
      cv::Mat work_edges;
      cv::Canny(work_img, work_edges, canny_low_thresh_,
                canny_low_thresh_ * canny_ratio_);
      work_edges &= mask;
      const EdgeStats st_work = computeEdgeStats(work_edges);
      edge_stats.wid_px = st_work.valid
        ? std::max(0.2, st_work.wid_px - width_bias_px_)
        : 0.0;
    }

    // ── 2. Tensor direccional de distancias (DT3V del paper) ────────────────
    cv_ext::BasicTimer bt;
    DistanceTransform dt;
    dt.enableParallelism(true);
    if (dist_threshold_ > 0.0) {
      dt.setDistThreshold(static_cast<float>(dist_threshold_));
    }
    if (edge_detector_type_ == "LSD") {
      auto ed = LSDEdgeDetectorUniquePtr(new LSDEdgeDetector());
      ed->setPyrNumLevels(2);
      ed->setMask(mask);
      dt.setEdgeDetector(std::move(ed));
    } else {
      auto ed = CannyEdgeDetectorUniquePtr(new CannyEdgeDetector());
      ed->setLowThreshold(canny_low_thresh_);
      ed->setRatio(canny_ratio_);
      ed->setMask(mask);
      dt.setEdgeDetector(std::move(ed));
    }
    cv_ext::ImageTensorPtr dist_map_tensor;
    dt.computeDistanceMapTensor(work_img, dist_map_tensor, num_directions_,
                                tensor_lambda_);
    matcher_->setInput(dist_map_tensor);
    const auto tensor_ms = bt.elapsedTimeMs();

    // ── 3. Hipótesis iniciales ──────────────────────────────────────────────
    std::vector<PoseHypothesis> hyps = buildHypotheses(bbox, edge_stats);
    if (hyps.empty()) {
      return {false, "no se pudieron generar hipotesis iniciales"};
    }

    // ── 4. Ranking barato por distancia chamfer direccional media ───────────
    for (auto & hyp : hyps) {
      hyp.score = matcher_->getAvgDistance(hyp.q, hyp.t);
    }
    std::sort(hyps.begin(), hyps.end(),
              [](const PoseHypothesis & a, const PoseHypothesis & b) {
                return a.score < b.score;
              });

    // ── 5. Refinado D2CO (Levenberg-Marquardt sobre el tensor) ──────────────
    bt.reset();
    PoseHypothesis best;
    int n_refine = std::min<int>(refine_top_k_, static_cast<int>(hyps.size()));
    // Atajo de tracking: con una semilla ya muy buena basta un refinado…
    // …pero cada full_search_every ciclos se explora igualmente, para no
    // quedar atrapados en un mínimo local que el tracking realimenta.
    ++cycle_count_;
    const bool full_search =
      (full_search_every_ > 0) && (cycle_count_ % full_search_every_ == 0);
    if (!full_search && hyps[0].score < tracking_fast_dist_) {
      n_refine = 1;
    }
    // Banda FÍSICA de profundidad: la proyección de un objeto de longitud L
    // solo puede ACORTARSE por escorzo, así que z_true ≤ z_est = fx·L/l_px.
    // Cualquier pose refinada fuera de z_est·[0.55, 1.15] es una solución
    // degenerada de la ambigüedad escala/profundidad y se descarta.
    const double z_band_ref = estimateDepth(bbox, edge_stats);
    const bool use_z_band = edge_stats.valid && edge_stats.theta_valid;
    const double z_lo = 0.55 * z_band_ref, z_hi = 1.15 * z_band_ref;

    // Selección del mejor candidato: la distancia media del tensor es casi
    // PLANA a lo largo del rayo (ambigüedad de escala de un cilindro sin
    // tapas visibles), así que se añade un coste de consistencia geométrica
    // con lo MEDIDO en la imagen: z frente a z*=fx·L·cosφ/l_px y centro
    // proyectado frente al centroide de los bordes.
    int n_descartes_banda = 0;
    double best_effective = std::numeric_limits<double>::max();
    for (int i = 0; i < n_refine; ++i) {
      Eigen::Quaterniond q = hyps[i].q;
      Eigen::Vector3d t = hyps[i].t;
      matcher_->refinePosition(q, t);
      if (use_z_band && (t.z() < z_lo || t.z() > z_hi)) {
        ++n_descartes_banda;
        continue;
      }
      const double refined_score = matcher_->getAvgDistance(q, t);
      double effective = refined_score;
      if (use_z_band && t.z() > 1e-6) {
        // Inclinación fuera de plano del eje refinado y z consistente con la
        // longitud medida de los bordes.
        const Eigen::Vector3d axis = q * Eigen::Vector3d::UnitZ();
        const double cphi = std::sqrt(
          std::max(1e-6, 1.0 - axis.z() * axis.z()));
        const double z_star =
          cam_model_.fx() * tube_length_ * cphi / edge_stats.len_px;
        effective += w_depth_ * std::abs(t.z() - z_star) / z_star;
        // Centro del modelo proyectado frente al centroide de los bordes.
        const double pu = cam_model_.fx() * t.x() / t.z() + cam_model_.cx();
        const double pv = cam_model_.fy() * t.y() / t.z() + cam_model_.cy();
        effective += w_center_ * std::hypot(pu - edge_stats.u,
                                            pv - edge_stats.v);
      }
      if (effective < best_effective) {
        best_effective = effective;
        best.q = q;
        best.t = t;
        best.score = refined_score;
      }
    }
    // ── 5b. Post-corrección por ANCHURA medida ──────────────────────────────
    // Sin tapas visibles, el LM se desliza por la variedad cosφ/z = cte
    // (compensa profundidad con inclinación: la longitud proyectada no
    // distingue). La separación de raíles SÍ es independiente del tilt:
    //     z = fx·Ø / w_px
    // Se desliza la pose por el rayo hasta esa profundidad y se recentra el
    // centro proyectado en el centroide de los bordes.
    // Se corrige SOLO cuando el ajuste es pobre (avg alto = rasgos
    // incompletos, p.ej. tapas sin contraste => deslizamiento por el valle).
    // Con avg bajo el optimizador ha anclado todos los rasgos y es más
    // preciso que la medida de anchura (±0.5 px => ±10% en z).
    if (use_z_band && best.score < std::numeric_limits<double>::max() &&
        best.score > width_correction_gate_ &&
        edge_stats.wid_px > 0.5) {
      const double w_px = edge_stats.wid_px;  // ya corregida de sesgo Canny
      const double z_w = cam_model_.fx() * tube_diameter_ / w_px;
      const double k = z_w / best.t.z();
      if (k > 0.6 && k < 1.6) {
        best.t *= k;  // deslizar por el rayo: misma proyección del centro
        // Recentrado lateral al centroide de los bordes.
        obj_model_->setModelView(best.q, best.t);
        std::vector<cv::Point2f> pp;
        obj_model_->projectRasterPoints(pp);
        double su = 0.0, sv = 0.0;
        int n_ok = 0;
        for (const auto & p : pp) {
          if (p.x < 0.0f && p.y < 0.0f) {
            continue;
          }
          su += p.x;
          sv += p.y;
          ++n_ok;
        }
        if (n_ok > 20) {
          const double du = edge_stats.u - su / n_ok;
          const double dv = edge_stats.v - sv / n_ok;
          best.t.x() += du * best.t.z() / cam_model_.fx();
          best.t.y() += dv * best.t.z() / cam_model_.fy();
        }
        best.score = matcher_->getAvgDistance(best.q, best.t);
        RCLCPP_INFO(get_logger(),
                    "%s correccion anchura: w=%.1fpx -> z=%.3fm (k=%.2f)",
                    log_prefix, w_px, z_w, k);
      }
    }

    const auto refine_ms = bt.elapsedTimeMs();
    if (n_descartes_banda > 0) {
      RCLCPP_INFO(get_logger(),
                  "%s %d pose(s) refinadas descartadas por la banda de "
                  "profundidad [%.2f, %.2f] m",
                  log_prefix, n_descartes_banda, z_lo, z_hi);
    }

    RCLCPP_INFO(get_logger(),
                "%s bbox(score=%.2f) bordes(%s c=%.0f,%.0f th=%.0fdeg "
                "l=%.0fpx) tensor=%lums refinado=%lums avg_dist=%.3f "
                "(umbral %.3f)",
                log_prefix, bbox_score,
                edge_stats.valid ? (edge_stats.theta_valid ? "ok" : "blob")
                                 : "NO",
                edge_stats.u, edge_stats.v,
                edge_stats.theta * 180.0 / M_PI, edge_stats.len_px,
                static_cast<unsigned long>(tensor_ms),
                static_cast<unsigned long>(refine_ms), best.score,
                max_avg_dist_);

    // ── 6. Rechazo de estimaciones de baja calidad ──────────────────────────
    if (best.score > max_avg_dist_) {
      return {false, "pose rechazada (avg_dist=" + std::to_string(best.score) +
                     " > " + std::to_string(max_avg_dist_) + ")"};
    }

    // ── 7. Publicación (mismo contrato que el nodo ICP) ─────────────────────
    publishPose(best, frame, img_stamp);
    {
      std::lock_guard<std::mutex> lk(mutex_);
      tracked_pose_ = best;
      tracked_pose_time_ = now();
      has_tracked_pose_ = true;
    }
    if (pub_overlay_) {
      publishOverlay(best, image, bbox, img_stamp);
    }

    return {true, "pose publicada (avg_dist=" + std::to_string(best.score) +
                  ")"};
  }

  // ══════════════════════════════════════════════════════════════════════════
  // HIPÓTESIS INICIALES — rayo del bbox + profundidad estimada + eje del tubo
  // ══════════════════════════════════════════════════════════════════════════

  /// Profundidad estimada del tamaño aparente: z_est = fx·L/l_px, donde l_px
  /// es la extensión de los bordes (PCA, robusta) o, en su defecto, el lado
  /// mayor del bbox. Por el escorzo, z_est es una cota SUPERIOR de la
  /// profundidad real (la proyección solo puede acortarse al inclinarse).
  double estimateDepth(const cv::Rect2d & bbox, const EdgeStats & st) const
  {
    const double extent_px = (st.valid && st.theta_valid)
                             ? st.len_px
                             : std::max(bbox.width, bbox.height) * work_scale_;
    double z_est = z_max_;
    if (extent_px > 1.0) {
      z_est = cam_model_.fx() * tube_length_ / extent_px;
    }
    return std::clamp(z_est, z_min_, z_max_);
  }

  std::vector<PoseHypothesis> buildHypotheses(const cv::Rect2d & bbox,
                                              const EdgeStats & st)
  {
    std::vector<PoseHypothesis> hyps;

    // ── Modo tracking: la última pose aceptada es la primera candidata ──────
    {
      std::lock_guard<std::mutex> lk(mutex_);
      if (has_tracked_pose_ &&
          (now() - tracked_pose_time_).seconds() < tracking_max_age_s_) {
        hyps.push_back(tracked_pose_);
      }
    }

    // ── Rayo de retroproyección del CENTRO DEL OBJETO ────────────────────────
    // Preferencia: centroide de los bordes (robusto frente a bbox de YOLO
    // holgados/descentrados). Fallback: centro del bbox.
    const double s = work_scale_;
    const double fx = cam_model_.fx(), fy = cam_model_.fy();
    const double cx = cam_model_.cx(), cy = cam_model_.cy();
    double u, v;
    if (st.valid) {
      u = st.u;
      v = st.v;
    } else {
      u = (bbox.x + bbox.width / 2.0) * s;
      v = (bbox.y + bbox.height / 2.0) * s;
    }
    Eigen::Vector3d ray((u - cx) / fx, (v - cy) / fy, 1.0);

    // ── Profundidad estimada por el tamaño aparente del tubo ────────────────
    // z_est es cota superior de la profundidad real (ver estimateDepth):
    // con bordes fiables se siembra DENTRO de la banda física [0.7,1.0]·z_est;
    // sin ellos se recurre al barrido amplio z_factors sobre el bbox.
    const double z_est = estimateDepth(bbox, st);
    std::vector<double> z_factors;
    if (st.valid && st.theta_valid) {
      z_factors = {0.7, 0.85, 1.0};
    } else {
      z_factors = z_factors_;
    }

    // ── Orientaciones candidatas del eje del tubo (eje Z del modelo) ────────
    // Preferencia: eje mayor PCA de los bordes (±20° de tolerancia).
    // Fallback: heurística por el aspecto del bbox. Un cilindro es simétrico
    // a rotaciones sobre su eje y a giros de 180°.
    std::vector<double> thetas;
    if (st.valid && st.theta_valid) {
      const double d20 = 20.0 * M_PI / 180.0;
      thetas = {st.theta - d20, st.theta, st.theta + d20};
    } else {
      const double aspect = bbox.width / std::max(bbox.height, 1e-6);
      if (aspect > 1.3) {
        thetas = {0.0};
      } else if (aspect < 1.0 / 1.3) {
        thetas = {M_PI / 2.0};
      } else {
        thetas = {0.0, M_PI / 4.0, M_PI / 2.0, 3.0 * M_PI / 4.0};
      }
    }

    for (const double zf : z_factors) {
      const double z = std::clamp(z_est * zf, z_min_, z_max_);
      const Eigen::Vector3d t = ray / ray.z() * z;
      for (const double theta : thetas) {
        for (const double tilt_deg : tilt_angles_deg_) {
          const double phi = tilt_deg * M_PI / 180.0;
          // Dirección del eje del tubo en el frame de cámara: componente en
          // el plano de imagen (theta) + inclinación fuera de plano (phi).
          const Eigen::Vector3d axis(std::cos(theta) * std::cos(phi),
                                     std::sin(theta) * std::cos(phi),
                                     std::sin(phi));
          PoseHypothesis hyp;
          hyp.q = Eigen::Quaterniond::FromTwoVectors(
            Eigen::Vector3d::UnitZ(), axis);
          hyp.t = t;
          hyps.push_back(hyp);
        }
      }
    }
    return hyps;
  }

  // ══════════════════════════════════════════════════════════════════════════
  // PUBLICACIÓN
  // ══════════════════════════════════════════════════════════════════════════

  void publishPose(const PoseHypothesis & pose, const std::string & frame,
                   const builtin_interfaces::msg::Time & stamp)
  {
    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp = stamp;
    ps.header.frame_id = frame;
    ps.pose.position.x = pose.t.x();
    ps.pose.position.y = pose.t.y();
    ps.pose.position.z = pose.t.z();
    Eigen::Quaterniond qn = pose.q.normalized();
    ps.pose.orientation.x = qn.x();
    ps.pose.orientation.y = qn.y();
    ps.pose.orientation.z = qn.z();
    ps.pose.orientation.w = qn.w();

    // ── Transformación opcional a otro frame (p.ej. world/base_link) ────────
    if (!output_frame_.empty() && output_frame_ != frame) {
      try {
        const auto tf = tf_buffer_->lookupTransform(
          output_frame_, frame, tf2::TimePointZero,
          tf2::durationFromSec(0.2));
        geometry_msgs::msg::PoseStamped out;
        tf2::doTransform(ps, out, tf);
        out.header.frame_id = output_frame_;
        out.header.stamp = stamp;
        ps = out;
      } catch (const tf2::TransformException & e) {
        RCLCPP_WARN(get_logger(),
                    "TF %s->%s no disponible (%s); se publica en %s.",
                    frame.c_str(), output_frame_.c_str(), e.what(),
                    frame.c_str());
      }
    }

    pub_pose_->publish(ps);

    if (pub_legacy_) {
      pub_legacy_pose_->publish(ps.pose);
    }

    if (broadcast_tf_) {
      geometry_msgs::msg::TransformStamped tf_msg;
      tf_msg.header = ps.header;
      tf_msg.child_frame_id = tf_child_frame_;
      tf_msg.transform.translation.x = ps.pose.position.x;
      tf_msg.transform.translation.y = ps.pose.position.y;
      tf_msg.transform.translation.z = ps.pose.position.z;
      tf_msg.transform.rotation = ps.pose.orientation;
      tf_broadcaster_->sendTransform(tf_msg);
    }

    RCLCPP_INFO(get_logger(), "Pose 6D publicada [%s]: x=%.3f y=%.3f z=%.3f",
                ps.header.frame_id.c_str(), ps.pose.position.x,
                ps.pose.position.y, ps.pose.position.z);
  }

  void publishOverlay(const PoseHypothesis & pose, const cv::Mat & full_img,
                      const cv::Rect2d & bbox,
                      const builtin_interfaces::msg::Time & stamp)
  {
    // Proyecta el raster del modelo en la pose estimada y lo dibuja sobre la
    // imagen ORIGINAL (las proyecciones están en coordenadas reescaladas).
    obj_model_->setModelView(pose.q, pose.t);
    std::vector<cv::Point2f> proj_pts;
    obj_model_->projectRasterPoints(proj_pts);

    cv::Mat overlay;
    cv::cvtColor(full_img, overlay, cv::COLOR_GRAY2BGR);
    const double inv_s = 1.0 / work_scale_;
    for (const auto & p : proj_pts) {
      if (p.x < 0.0f && p.y < 0.0f) {
        continue;  // puntos no proyectables marcados (-1,-1) por la libreria
      }
      const cv::Point pt(static_cast<int>(p.x * inv_s),
                         static_cast<int>(p.y * inv_s));
      if (pt.x < 0 || pt.y < 0 || pt.x >= overlay.cols ||
          pt.y >= overlay.rows) {
        continue;
      }
      cv::circle(overlay, pt, 1, cv::Scalar(0, 255, 255), -1);
    }
    cv::rectangle(overlay,
                  cv::Point(static_cast<int>(bbox.x),
                            static_cast<int>(bbox.y)),
                  cv::Point(static_cast<int>(bbox.x + bbox.width),
                            static_cast<int>(bbox.y + bbox.height)),
                  cv::Scalar(0, 0, 255), 2);

    cv_bridge::CvImage out(std_msgs::msg::Header(), "bgr8", overlay);
    out.header.stamp = stamp;
    pub_overlay_image_->publish(*out.toImageMsg());
  }

  // ══════════════════════════════════════════════════════════════════════════
  // MIEMBROS
  // ══════════════════════════════════════════════════════════════════════════

  // Parámetros
  std::string image_topic_, output_frame_, tf_child_frame_, stl_path_;
  std::string edge_detector_type_, target_class_;
  bool broadcast_tf_ = true, pub_legacy_ = false, pub_overlay_ = true;
  double step_meters_ = 0.001, image_scale_ = 0.5, tensor_lambda_ = 6.0;
  int num_directions_ = 60, optimizer_iters_ = 50;
  int canny_low_thresh_ = 40, canny_ratio_ = 2;
  double dist_threshold_ = -1.0, min_score_ = 0.3, bbox_margin_px_ = 24.0;
  double det_max_age_s_ = 1.0, tube_length_ = 0.186;
  double tube_diameter_ = 0.0208, width_bias_px_ = 1.0;
  double width_correction_gate_ = 1.5;
  double z_min_ = 0.3, z_max_ = 1.5, max_avg_dist_ = 4.5;
  double w_depth_ = 5.0, w_center_ = 0.05;
  double tracking_max_age_s_ = 3.0, tracking_fast_dist_ = 1.0;
  std::vector<double> z_factors_, tilt_angles_deg_;
  int refine_top_k_ = 3, full_search_every_ = 5;
  long cycle_count_ = 0;

  // Estado cacheado por los callbacks
  std::mutex mutex_;
  bool have_camera_info_ = false;
  sensor_msgs::msg::CameraInfo camera_info_;
  cv::Mat last_image_;
  builtin_interfaces::msg::Time last_image_stamp_;
  std::string image_frame_;
  cv::Rect2d last_bbox_;
  double last_bbox_score_ = -1.0;
  rclcpp::Time last_detection_time_;
  PoseHypothesis tracked_pose_;
  rclcpp::Time tracked_pose_time_;
  bool has_tracked_pose_ = false;
  bool busy_ = false;

  // Objetos D2CO (se crean una única vez; el GL queda ligado a este hilo)
  bool d2co_ready_ = false;
  double work_scale_ = 1.0;  // px de la imagen de trabajo / px originales
  cv_ext::PinholeCameraModel cam_model_;
  RasterObjectModel3DPtr obj_model_;
  std::shared_ptr<DirectionalChamferMatching> matcher_;

  // ROS
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr
    sub_detections_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
    sub_camera_info_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;
  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pub_legacy_pose_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_overlay_image_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_estimate_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // Ejecutor de un solo hilo: todas las callbacks (y por tanto todo el uso
  // del contexto OpenGL de D2CO) viven en el mismo hilo.
  auto node = std::make_shared<D2coPoseEstimatorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
