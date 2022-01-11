```shell
#!/bin/bash

set -e

if [ $# -eq 0 ] || [ $# -gt 2 ]
then
  echo "wrong number of parameters provided"
  echo "example: ./map_pipeline_local.sh db 0.0.1"
  echo "example: ./map_pipeline_local.sh file"
  exit 1;
fi  

if [ $# -eq 1 ] && [ $1 != "file" ]
then
  echo "wrong parameters provided"
  echo "example: ./map_pipeline_local.sh db 0.0.1"
  echo "example: ./map_pipeline_local.sh file"
  exit 2;
fi  

if [ $# -eq 2 ] && [ $1 != "db" ]
then 
  echo "wrong parameters provided"
  echo "example: ./map_pipeline_local.sh db 0.0.1"
  echo "example: ./map_pipeline_local.sh file"
  exit 3; 
fi

CONTAINER_NAME="map_pipeline_local_storage"
DOCKER_REGISTRY_URL="allride-registry.cn-shanghai.cr.aliyuncs.com"

# function define
docker_run() {
  docker run -itd -p 5432:5432 --name $CONTAINER_NAME $DOCKER_REGISTRY_URL/map/map_storage:vempty
  until docker exec $CONTAINER_NAME psql; do
    echo "local map db is starting up"
    sleep 10
  done
  echo "local map db started"  
}

exist_clear_nonexist_create() {
  if [ ! -d "$1" ];
  then
    mkdir $1
  else
    rm -rf $1/*
  fi
}

echo "pipeline mode: $1"
PIPELINE_MODE="$1"
PIPELINE_CONF_FOLDER_NAME=""
if [ $PIPELINE_MODE == "db" ]
then
  echo "-----------------------------------------------------------------------------"
  echo "REMINDER: please docker login and compile project before running this script"
  # check if old container is running
  {
    ls_info=$(docker container ls -a | grep $CONTAINER_NAME)
    echo "old version of $CONTAINER_NAME container exist, to stop and remove"
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    echo "old $CONTAINER_NAME stopped and rem#!/bin/bash

set -e

if [ $# -eq 0 ] || [ $# -gt 2 ]
then
  echo "wrong number of parameters provided"
  echo "example: ./map_pipeline_local.sh db 0.0.1"
  echo "example: ./map_pipeline_local.sh file"
  exit 1;
fi

if [ $# -eq 1 ] && [ $1 != "file" ]
then
  echo "wrong parameters provided"
  echo "example: ./map_pipeline_local.sh db 0.0.1"
  echo "example: ./map_pipeline_local.sh file"
  exit 2;
fi

if [ $# -eq 2 ] && [ $1 != "db" ]
then
  echo "wrong parameters provided"
<c/map/processor/map_pipeline_local.sh" 205L, 8415C           1,1           Top
oved"
  } || {
    if [ -z "$ls_info" ]; then 
      echo "$CONTAINER_NAME is not running, no need to remove, continue"
    fi
  }
  echo "to pull docker image and run local db"
  docker pull $DOCKER_REGISTRY_URL/map/map_storage:vempty
  docker_run
  PIPELINE_CONF_FOLDER_NAME="data_pipeline_db"
elif [ $PIPELINE_MODE == "file" ]
then
  PIPELINE_CONF_FOLDER_NAME="data_pipeline"
else
  echo "pipeline only support db or file"
  exit 1
fi

# pipeline start from here
echo "-----------------------------------------------------------------------------"
echo "---------------------------PIPILINE START----------------------------------"
LOCAL_DATA_PATH="/opt/allride/data/map"
echo "REMINDER: please put the routing_pairs.pb.txt file under $LOCAL_DATA_PATH"
MAP_BUILDER_PATH="../../../devel/lib"
CONF_PATH="$LOCAL_DATA_PATH/processor"
exist_clear_nonexist_create $CONF_PATH
cp -r ../conf/processor $LOCAL_DATA_PATH/
echo "---------------------------BASE_MAP_LOADER---------------------------------"
exist_clear_nonexist_create $LOCAL_DATA_PATH/road_map_raw
$MAP_BUILDER_PATH/map/map_builder $CONF_PATH/conversion/base_map_to_file.cfg

echo "--------------------------ROAD_MAP_GEOMETRY_RESAMPLE-----------------------"
exist_clear_nonexist_create $LOCAL_DATA_PATH/road_map_geometry_resampled
$MAP_BUILDER_PATH/map/map_builder $CONF_PATH/$PIPELINE_CONF_FOLDER_NAME/geometry_resample.cfg

echo "--------------------------ROAD_MAP_LANE_TOPOLOGY_SAMPLE--------------------"
exist_clear_nonexist_create $LOCAL_DATA_PATH/road_map_lane_topology_processed
$MAP_BUILDER_PATH/map/map_builder $CONF_PATH/$PIPELINE_CONF_FOLDER_NAME/lane_topology.cfg

echo "--------------------------HDMapValidator2d---------------------------------"
$MAP_BUILDER_PATH/map/map_builder $CONF_PATH/validation/hdmap_validator2d.cfg
if [ -s $LOCAL_DATA_PATH/hdmap_validator2d_error_info.txt ]
then
  echo "HDMapValidator2d found errors, pipeline stopped"
  echo "error file path : $LOCAL_DATA_PATH/hdmap_validator2d_error_info.txt"
  exit 2
fi

echo "--------------------------ROAD_MAP_TILE_GROUND_SAMPLE----------------------"
exist_clear_nonexist_create $LOCAL_DATA_PATH/road_map_tile_ground_sample_processed
$MAP_BUILDER_PATH/map/map_builder $CONF_PATH/$PIPELINE_CONF_FOLDER_NAME/tile_ground_sample.cfg

echo "--------------------------ROAD_MAP_LANE_GROUND_SAMPLE----------------------"
exist_clear_nonexist_create $LOCAL_DATA_PATH/road_map_lane_ground_sample_processed
$MAP_BUILDER_PATH/map/map_builder $CONF_PATH/$PIPELINE_CONF_FOLDER_NAME/lane_ground_sample.cfg
if [ -s $LOCAL_DATA_PATH/lane_ground_sample_error_info.txt ]
then
  echo "lane ground sample missing"
  echo "error file path : $LOCAL_DATA_PATH/lane_ground_sample_error_info.txt"
fi

echo "--------------------------ROAD_MAP_OTHER_GROUND_SAMPLE-----------------------"
exist_clear_nonexist_create $LOCAL_DATA_PATH/road_map_other_ground_sample_processed
$MAP_BUILDER_PATH/map/map_builder $CONF_PATH/$PIPELINE_CONF_FOLDER_NAME/other_ground_sample.cfg
if [ -s $LOCAL_DATA_PATH/other_ground_sample_error_info.txt ]
then
  echo "other ground sample missing"
  echo "error file path : $LOCAL_DATA_PATH/other_ground_sample_error_info.txt"
fi

echo "--------------------------ROAD_MAP_HEIGHT------------------------------------"
exist_clear_nonexist_create $LOCAL_DATA_PATH/road_map_height_processed
$MAP_BUILDER_PATH/map/map_builder $CONF_PATH/$PIPELINE_CONF_FOLDER_NAME/height.cfg

echo "--------------------------ROAD_MAP_LANE_OVERLAP------------------------------"
exist_clear_nonexist_create $LOCAL_DATA_PATH/road_map_lane_overlap_processed
$MAP_BUILDER_PATH/map/map_builder $CONF_PATH/$PIPELINE_CONF_FOLDER_NAME/lane_overlap.cfg

echo "--------------------------ROAD_MAP_CONNECTION_TOPOLOGY-----------------------"
exist_clear_nonexist_create $LOCAL_DATA_PATH/road_map_connection_topology_processed
$MAP_BUILDER_PATH/map/map_builder $CONF_PATH/$PIPELINE_CONF_FOLDER_NAME/connection_topology.cfg

echo "--------------------------ROAD_MAP_LANE_WIDTH--------------------------------"
exist_clear_nonexist_create $LOCAL_DATA_PATH/road_map_lane_width_processed
$MAP_BUILDER_PATH/map/map_builder $CONF_PATH/$PIPELINE_CONF_FOLDER_NAME/lane_width.cfg

echo "--------------------------ROAD_MAP_RELATION----------------------------------"
exist_clear_nonexist_create $LOCAL_DATA_PATH/road_map_relation_processed
$MAP_BUILDER_PATH/map/map_builder $CONF_PATH/$PIPELINE_CONF_FOLDER_NAME/relation.cfg

echo "--------------------------HDMapValidator3d-----------------------------------"
$MAP_BUILDER_PATH/map/map_builder $CONF_PATH/validation/hdmap_validator3d.cfg
routingFailedAmount=$(grep -rn "routing" $LOCAL_DATA_PATH/hdmap_validator3d_error_info.txt | wc -l)
declare -i fileLineAmount
fileLineAmount=`cat $LOCAL_DATA_PATH/hdmap_validator3d_error_info.txt | wc -l`
if [ $routingFailedAmount -gt 0 ]
then
    echo "routing error occurred, pipeline stopped"
    echo "error file path : $LOCAL_DATA_PATH/hdmap_validator3d_error_info.txt"
    exit 3
elif [ $fileLineAmount -gt 6 ]
then
  echo "HDMapValidator3d has warnings or errors"
  echo "error file path : $LOCAL_DATA_PATH/hdmap_validator3d_error_info.txt"
fi

echo "--------------------------ROAD_MAP_CLEAN_UP----------------------------------"
exist_clear_nonexist_create $LOCAL_DATA_PATH/road_map_clearance
$MAP_BUILDER_PATH/map/map_builder $CONF_PATH/$PIPELINE_CONF_FOLDER_NAME/clean_up.cfg

# echo "--------------------------ROAD_MAP_REFLINE-----------------------------------"
# exist_clear_nonexist_create $LOCAL_DATA_PATH/road_map_smoothed
# $MAP_BUILDER_PATH/map/map_builder $CONF_PATH/$PIPELINE_CONF_FOLDER_NAME/refline.cfg

echo "--------------------------ROAD_MAP_ROI---------------------------------------"
exist_clear_nonexist_create $LOCAL_DATA_PATH/roi
$MAP_BUILDER_PATH/map/map_builder $CONF_PATH/$PIPELINE_CONF_FOLDER_NAME/roi.cfg

echo "--------------------------ENVIRONMENT_MAP_EDGE-------------------------------"
exist_clear_nonexist_create $LOCAL_DATA_PATH/environment_map
$MAP_BUILDER_PATH/map/map_builder $CONF_PATH/$PIPELINE_CONF_FOLDER_NAME/edge.cfg

echo "--------------------------PIPELINE END---------------------------------------"

if [ $PIPELINE_MODE == "db" ]
then
  echo "to extract raw file"
  $MAP_BUILDER_PATH/map/map_builder $CONF_PATH/conversion/db_to_raw.cfg
  echo "to commit db container"
  docker commit $CONTAINER_NAME $DOCKER_REGISTRY_URL/map/map_storage:v$2
  echo "to extract raw file"
  $MAP_BUILDER_PATH/map/map_builder $CONF_PATH/conversion/db_to_raw.cfg
  echo "to stop and rm db container"
  docker stop $CONTAINER_NAME
  docker rm $CONTAINER_NAME

else
  echo "to extract raw file"
  $MAP_BUILDER_PATH/map/map_builder $CONF_PATH/conversion/file_to_raw.cfg
fi

echo "-----------------------------finish--------------------------------------"
```