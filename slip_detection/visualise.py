import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


data_dir = "/home/user/Robotics/CDNA/CDNA/slip_detection/models/20210125-153921-CDNA-16/"  # 20210124-143736-CDNA-16

# losses_train = np.load(data_dir + "training-global_losses.npy")
# losses_valid = np.load(data_dir + "training-global_losses_valid.npy")
# psnr = np.load(data_dir + "training-global_psnr_all.npy")

original_images = np.load(data_dir + "original_images.npy")
predicted_images = np.load(data_dir + "predicted_images.npy")

original_images = np.asarray(original_images).astype(np.uint8)
predicted_images = np.asarray(predicted_images).astype(np.uint8)

original_images = np.asarray([img[0].T for img in original_images])
predicted_images = np.asarray([img.T for img in predicted_images])

# losses_train_means = []
# losses_valid_means = []
# psnr_means = []
# for i in range(0, len(losses_train)):
#     losses_train_means.append(np.mean(losses_train[i]))
#     losses_valid_means.append(np.mean(losses_valid[i]))
#     psnr_means.append(np.mean(psnr[i]))

# losses_train_means = losses_train_means[0:20]
# losses_valid_means = losses_valid_means[0:20]
# psnr_means = psnr_means[0:50]

# plt.plot([i for i in range(0, len(losses_train_means))], losses_train_means, c='r')
# plt.plot([i for i in range(0, len(losses_valid_means))], losses_valid_means, c='g')
# plt.plot([i for i in range(0, len(psnr_means))], psnr_means, c='g')
# plt.ylabel('loss == MSE')
# plt.xlabel('epoch')
# plt.show()

original_images = original_images[1:]
# for i in range(0, len(original_images)):
#     for ov, pv in zip(original_images[i], predicted_images[i]):
#         print(abs(ov - pv))
# print(len(predicted_images))

print(np.mean(np.subtract(predicted_images, original_images)))

# fig = plt.figure()

# print(predicted_images[0]*255)
# print(original_images[0]*255)

f, axarr = plt.subplots(2,9)
axarr[0,0].imshow(predicted_images[0])
axarr[0,0].set_title("0: P: {0}".format(int(np.sum(predicted_images[0]))))
axarr[0,1].imshow(predicted_images[1])
axarr[0,1].set_title("1: P: {0}".format(int(np.sum(predicted_images[1]))))
axarr[0,2].imshow(predicted_images[2])
axarr[0,2].set_title("2: P: {0}".format(int(np.sum(predicted_images[2]))))
axarr[0,3].imshow(predicted_images[3])
axarr[0,3].set_title("3: P: {0}".format(int(np.sum(predicted_images[3]))))
axarr[0,4].imshow(predicted_images[4])
axarr[0,4].set_title("4: P: {0}".format(int(np.sum(predicted_images[4]))))
axarr[0,5].imshow(predicted_images[5])
axarr[0,5].set_title("5: P: {0}".format(int(np.sum(predicted_images[5]))))
axarr[0,6].imshow(predicted_images[6])
axarr[0,6].set_title("6: P: {0}".format(int(np.sum(predicted_images[6]))))
axarr[0,7].imshow(predicted_images[7])
axarr[0,7].set_title("7: P: {0}".format(int(np.sum(predicted_images[7]))))
axarr[0,8].imshow(predicted_images[8])
axarr[0,8].set_title("8: P: {0}".format(int(np.sum(predicted_images[8]))))
axarr[0,0].set_ylabel("Pred", rotation=90, size='large')

axarr[1,0].imshow(original_images[0])
axarr[1,0].set_title("0: GT: {0}".format(int(np.sum(original_images[0]))))
axarr[1,1].imshow(original_images[1])
axarr[1,1].set_title("1: GT: {0}".format(int(np.sum(original_images[1]))))
axarr[1,2].imshow(original_images[2])
axarr[1,2].set_title("2: GT: {0}".format(int(np.sum(original_images[2]))))
axarr[1,3].imshow(original_images[3])
axarr[1,3].set_title("3: GT: {0}".format(int(np.sum(original_images[3]))))
axarr[1,4].imshow(original_images[4])
axarr[1,4].set_title("4: GT: {0}".format(int(np.sum(original_images[4]))))
axarr[1,5].imshow(original_images[5])
axarr[1,5].set_title("5: GT: {0}".format(int(np.sum(original_images[5]))))
axarr[1,6].imshow(original_images[6])
axarr[1,6].set_title("6: GT: {0}".format(int(np.sum(original_images[6]))))
axarr[1,7].imshow(original_images[7])
axarr[1,7].set_title("7: GT: {0}".format(int(np.sum(original_images[7]))))
axarr[1,8].imshow(original_images[8])
axarr[1,8].set_title("8: GT: {0}".format(int(np.sum(original_images[8]))))
axarr[1,0].set_ylabel("GT", rotation=90, size='large')

plt.show()
# img = Image.fromarray(predicted_images[0], 'RGB')
# img.show()
