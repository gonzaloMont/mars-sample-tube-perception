#pragma once

#define OBJECT_VERTEX_SHADER_CODE(VERSION) \
"\
#version "+VERSION+"\
\n in vec3 vertex_pos; \
uniform mat4 rt_persp;\
void main()\
{\
  gl_Position =  rt_persp * vec4(vertex_pos,1);\
}\
"

#define COLORED_OBJECT_VERTEX_SHADER_CODE(VERSION) \
"\
#version "+VERSION+"\
\n in vec3 vertex_pos; \
in vec3 vertex_color; \
out vec3 fragment_color; \
uniform mat4 rt_persp; \
void main()\
{\
  gl_Position =  rt_persp * vec4(vertex_pos,1);\
  fragment_color = vertex_color; \
}\
"

#define COLORED_OBJECT_FRAGMENT_SHADER_CODE(VERSION) \
"\
#version "+VERSION+"\
\n in vec3 fragment_color; \
out vec4 fragment_out; \
void main()\
{\
  fragment_out = vec4(fragment_color, 1);\
}\
"

#define SHADED_OBJECT_VERTEX_SHADER_CODE(VERSION) \
"\
#version "+VERSION+"\
\n in vec3 vertex_pos; \
in vec3 vertex_color; \
in vec3 vertex_normal; \
out vec3 fragment_color; \
out vec3 pos_world; \
out vec3 normal_cam; \
out vec3 eye_dir_cam; \
out vec3 light_dir_cam; \
uniform mat4 rt_persp; \
uniform mat4 model_view; \
uniform vec3 light_pos_world; \
void main()\
{\
  gl_Position =  rt_persp * vec4(vertex_pos,1); \
  pos_world = vertex_pos;\
  vec3 pos_cam = ( model_view * vec4(vertex_pos,1)).xyz; \
  eye_dir_cam = vec3(0,0,0) - pos_cam; \
  vec3 light_pos_cam = ( model_view * vec4(light_pos_world,1)).xyz; \
  light_dir_cam = light_pos_cam + eye_dir_cam; \
  normal_cam = ( model_view * vec4(vertex_normal,0)).xyz; \
  fragment_color = vertex_color; \
}\
"

/*
#define SHADED_OBJECT_VERTEX_SHADER_CODE(VERSION) \
"\
#version "+VERSION+"\
\n in vec3 vertex_pos; \
in vec3 vertex_color; \
in vec3 vertex_normal; \
out vec3 fragment_color; \
out vec3 pos_world; \
out vec3 normal_cam; \
out vec3 eye_dir_cam; \
out vec3 light_dir_cam; \
uniform mat4 rt_persp;\
uniform mat4 model_view;\
uniform mat4 model_mat;\
uniform vec3 light_pos_world;\
void main()\
{\
  gl_Position =  rt_persp * vec4(vertex_pos,1);\
  pos_world = (model_mat * vec4(vertex_pos,1)).xyz;\
  vec3 pos_cam = ( model_view * model_mat * vec4(vertex_pos,1)).xyz;\
  eye_dir_cam = vec3(0,0,0) - pos_cam;\
  vec3 light_pos_cam = ( model_view * vec4(light_pos_world,1)).xyz;\
  light_dir_cam = LightPosition_cameraspace + eye_dir_cam;\
  normal_cam = ( model_view * model_mat * vec4(vertex_normal,0)).xyz;\
  fragment_color = vertex_color; \
}\
"
*/

#define SHADED_OBJECT_FRAGMENT_SHADER_CODE(VERSION) \
"\
#version "+VERSION+"\
\n in vec3 fragment_color; \
in vec3 pos_world; \
in vec3 normal_cam; \
in vec3 eye_dir_cam; \
in vec3 light_dir_cam; \
uniform vec3 light_pos_world; \
out vec4 fragment_out; \
void main()\
{\
  vec3 light_color = vec3(0.8,0.8,0.8); \
  float light_power = 1.0f; \
  vec3 material_ambient_color = vec3(0.1,0.1,0.1) * fragment_color; \
  vec3 material_specular_color = vec3(1.0,1.0,1.0); \
  float distance = length( light_pos_world - pos_world ); \
  vec3 n = normalize( normal_cam ); \
  vec3 l = normalize( light_dir_cam ); \
  float cos_theta = clamp( dot( n,l ), 0,1 ); \
  vec3 E = normalize(eye_dir_cam); \
  vec3 R = reflect(-l,n); \
  float cos_alpha = clamp( dot( E,R ), 0,1 ); \
  fragment_out = vec4( (material_ambient_color + \
                      fragment_color * light_color * light_power * cos_theta + \
                      material_specular_color * light_color * light_power * pow(cos_alpha,5)  ), 1 ); \
}\
"

/*
#define SHADED_OBJECT_FRAGMENT_SHADER_CODE(VERSION) \
"\
#version "+VERSION+"\
\n in vec3 fragment_color; \
in vec3 pos_world; \
in vec3 normal_cam; \
in vec3 eye_dir_cam; \
in vec3 light_dir_cam; \
uniform vec3 light_pos_world; \
out vec4 fragment_out; \
void main()\
{\
  vec3 light_color = vec3(1,1,1); \
  float light_power = 50.0f; \
  vec3 material_ambient_color = vec3(0.1,0.1,0.1) * fragment_color; \
  vec3 material_specular_color = vec3(0.3,0.3,0.3); \
  float distance = length( light_pos_world - pos_world ); \
  vec3 n = normalize( normal_cam ); \
  vec3 l = normalize( light_dir_cam ); \
  float cos_theta = clamp( dot( n,l ), 0,1 ); \
  vec3 E = normalize(eye_dir_cam); \
  vec3 R = reflect(-l,n); \
  float cos_alpha = clamp( dot( E,R ), 0,1 ); \
  fragment_out = vec4( (material_ambient_color + \
                      fragment_color * light_color * light_power * cos_theta / (distance*distance) + \
                      material_specular_color * light_color * light_power * pow(cos_alpha,5) / (distance*distance) ), 1 ); \
}\
"
*/
