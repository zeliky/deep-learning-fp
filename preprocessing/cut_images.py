
target_size = model_options
for user_id in range(100,140):
  print(f"splitting files -  user: {user_id}")
  targe_size = model_options.image_height, model_options.image_width
  uds = UserDataset(user_id )
  uds.store_sub_lines(targe_size)
drive.flush_and_unmount()