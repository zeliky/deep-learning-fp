from sklearn import   metrics

def show_confusion_matrix(all_labels, all_predictions):
  print("Confusion matrix:\n")
  cmatrix = metrics.confusion_matrix(all_labels, all_predictions)
  for  i,r in enumerate(cmatrix):
    if i==0:
      line = ''.ljust(5)
      for user in range(0,len(r)):
        line += str(user).ljust(3)
      print( line)

    line = str(i).ljust(5)
    for c in r:
      if c<2:
        c= ' '.ljust(3)
      line += str(c).ljust(3)
    print( line)



user_ids= [3,4,8]
input_shape = (model_options.image_height, model_options.image_width, 1)
print(input_shape)
model = OneLetterClassifierModel().get_model(num_classes=model_options.num_classes , input_shape=input_shape)
model.summary()
model.load_weights(MODEL_CHECKPOINT_PATH+'ft-one-etter-lassifier-model-04-0.09.hdf5')

testing_gen = LettersGenerator(MODE_TEST, user_ids, model_options, model_options.num_classes)
all_predictions, all_labels = [], []
for i, (batch_x, labels) in enumerate(testing_gen):
  predict_values = model.predict(batch_x)
  all_labels += np.argmax(labels, axis=1).tolist()
  all_predictions += np.argmax(predict_values, axis=1).tolist()

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
accuracy = sum(all_predictions == all_labels) / len(all_labels)
print(f"Accuracy: {accuracy:.4f}")

print("Logistic regression using  features cross validation:\n%s\n" % (metrics.classification_report(all_labels,all_predictions)))

show_confusion_matrix(all_labels, all_predictions)


user_ids = [3,4,8]
testing_gen = LettersGenerator(MODE_TEST, user_ids, model_options, model_options.num_classes)

layer_outputs = []
layer_names = []
for layer in model.layers:
    if isinstance(layer, (Conv2D)):
        layer_outputs.append(layer.output)
        layer_names.append(layer.name)
activation_model = Model(inputs=model.input, outputs=layer_outputs)
test_images, lables = testing_gen.__getitem__(1)
activations = activation_model.predict(test_images[3:4])


testing_gen = LettersGenerator(MODE_TEST, user_ids, model_options, model_options.num_classes)
test_images, lables = testing_gen.__getitem__(1)
show_sequence(test_images)




import matplotlib.pyplot as plt

im_index =80
image = test_images[im_index]
plt.matshow(image, cmap='gray')
activations = activation_model.predict(test_images[im_index:im_index+1])




images_per_row = 8
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    print(n_features)
    if (n_features<8):
      continue
    #print(n_features)
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    #print(size)
    #print(n_cols)
    display_grid = np.zeros(((size + 1) * n_cols - 1,
                             images_per_row * (size + 1) - 1))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_index = col * images_per_row + row
            channel_image = layer_activation[0, :, :, channel_index].copy()
            if channel_image.sum() != 0:
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype("uint8")
            display_grid[
                col * (size + 1): (col + 1) * size + col,
                row * (size + 1) : (row + 1) * size + row] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.axis("off")
    plt.imshow(display_grid, aspect="auto", cmap="gray")