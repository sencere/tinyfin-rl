make play_dqn

cd environments/maze_rooms
make clean
make libmaze_rooms.so libmaze_rooms_render.so
cd ../..

./build/play_dqn --env environments/maze_rooms/libmaze_rooms.so \
  --render environments/maze_rooms/libmaze_rooms_render.so \
  --load runs/maze_dqn --fps 10

