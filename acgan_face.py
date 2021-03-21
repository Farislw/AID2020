import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import glob
image_path=glob.glob('path')
np.random.seed(2020)
np.random.shuffle(image_path)
labels=[p.split('\\')[1] for p in image_path]#这里移植的时候可能不是'//'
name2cls=dict((i,name)for i,name in enumerate(np.unique(labels)))#{0:'man',1:'woman'}
cls2name=dict((name,i)for i,name in enumerate(np.unique(labels)))#{'man':0,'woman':1}
labels=[cls2name.get(name)for name in labels]
image_path=np.array(image_path)
labels=np.array(labels)
def load_images(path):
    """
    因为我们在这里做了图像增强，所以生成的图像会根据样本也会可能产生相应的扭曲，
    在实际过程中可以把resize去掉，直接crop或者先padding再crop，就会减少扭曲
    :param path:
    :return:
    """
    img=tf.io.read_file(path)
    img=tf.image.decode_jpeg(img)
    img=tf.image.resize(img,(80,80))
    img=tf.image.random_crop(img,[64,64,3])
    img=tf.image.random_flip_left_right(img)
    img=img/127.5-1
    return img
img_dataset=tf.data.Dataset.from_tensor_slices(image_path)
img_dataset=img_dataset.map(load_images)#img_dataset:(None,64,64,3),types:tf.float32
label_dataset=tf.data.Dataset.from_tensor_slices(labels)
dataset=tf.data.Dataset.zip((img_dataset,label_dataset))#dataset shapes:((64,64,3).()),types:(yf.float32,tf.int32)
BATCH_SIZE=16#根据显存弄合适的
noise_dim=50
image_count=len(image_path)
dataset=dataset.shuffle(300).batch(BATCH_SIZE)#因为前面已经做过乱序处理，所以这里可以小范围乱序


def generate():
    seed=tf.keras.layers.Input(shape=((noise_dim)))
    label=tf.keras.layers.Input(shape=(()))#代表单个的值
    x=tf.keras.layers.Embedding(2,50,input_length=1)(label)
    x=tf.keras.layers.concatenate([seed,x])
    print(x.shape)
    x=tf.keras.layers.Dense(4*4*64*8,use_bias=False)(x)
    x=tf.keras.layers.Reshape((4,4,64*8))(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.ReLU()(x)

    x=tf.keras.layers.Conv2DTranspose(64*4,(3,3),strides=(2,2),use_bias=False)(x)#因为默认是valid，根据公式，存疑（？）
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.ReLU()(x)#(8*8*(64*4))

    x = layers.Conv2DTranspose(64*2, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)  # 16*16*(64*2)

    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)  # 32*32*64


    x=tf.keras.layers.Conv2DTranspose(3,(3,3),strides=(2,2),padding='same',use_bias=False)(x)
    x=layers.Activation('tanh')(x)#64*64*3,因为生成的图像是彩色图像三通道
    model=tf.keras.Model(inputs=[seed,label],outputs=x)

    return model
gen=generate()
def discriminator_model():
    image=tf.keras.Input(shape=((64,64,3)))


    x=tf.keras.layers.Conv2D(64,(3,3),strides=(2,2),padding='same',use_bias=False)(image)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.LeakyReLU()(x)#因为relu在0后就没有了，leaky会在0后保留部分输出，为了让判别器能继续对抗，
    x=tf.keras.layers.Dropout(0.5)(x)#为了让辨别器不那么强，一开始，不然对抗效果不好，32*32*64

    x = tf.keras.layers.Conv2D(64*2, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)  # 因为relu在0后就没有了，leaky会在0后保留部分输出，为了让判别器能继续对抗，
    x = tf.keras.layers.Dropout(0.5)(x)  # 为了让辨别器不那么强，一开始，不然对抗效果不好，16*16*（64*2）

    x=tf.keras.layers.Conv2D(64*4,(3,3),strides=(2,2),padding='same',use_bias=False)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.LeakyReLU()(x)
    x=tf.keras.layers.Dropout(0.5)(x)#8*8*(64*4)

    x = tf.keras.layers.Conv2D(64 * 8, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)  # 4*4*(64*8)

    x=tf.keras.layers.Flatten()(x)
    x1=tf.keras.layers.Dense(1)(x)#真假输出
    x2=tf.keras.layers.Dense(2)(x)#分类输出
    model=tf.keras.Model(inputs=image,outputs=[x1,x2])
    return model
dis=discriminator_model()
bce=tf.keras.losses.BinaryCrossentropy(from_logits=True)
cce=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def dis_loss(real_output,real_cls_out,fake_output,label):
    """

    :param real_output:
    :param real_cls_out:真实图片的类别输出
    :param fake_output:
    :param label: 真正的类别
    :return:
    """
    real_loss=bce(tf.ones_like(real_output),fake_output)
    fake_loss=bce(tf.zeros_like(fake_output),fake_output)
    cat_loss=cce(label,real_cls_out)
    return real_loss+fake_loss+cat_loss
def generator_loss(fake_output,fake_cls_out,label):
    fake_loss = bce(tf.ones_like(fake_output), fake_output)
    cat_loss=cce(label,fake_cls_out)
    return fake_loss+cat_loss
gen_opt=tf.keras.optimizers.Adam(1e-5)
dis_opt=tf.keras.optimizers.Adam(1e-5)


@tf.function
def train_step(images, labels):
    batchsize = labels.shape[0]
    noise = tf.random.normal([batchsize, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen((noise, labels), training=True)

        real_output,real_cls_out = dis(images, training=True)
        fake_output,fake_cls_out = dis(generated_images, training=True)

        gen_loss = generator_loss(fake_output,fake_cls_out,labels)
        disc_loss = dis_loss(real_output, real_cls_out,fake_output,labels)

    gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, dis.trainable_variables)

    gen_opt.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    dis_opt.apply_gradients(zip(gradients_of_discriminator, dis.trainable_variables))
noise_dim = 50
num = 10
noise_seed = tf.random.normal([num, noise_dim])
cat_seed = np.random.randint(0, 10, size=(num, 1))
print(cat_seed.T)
def generate_images(model, test_noise_input, test_cat_input, epoch):
    print('Epoch:', epoch+1)
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    predictions = model((test_noise_input, test_cat_input), training=False)
    fig = plt.figure(figsize=(10, 1))

    for i in range(predictions.shape[0]):
        plt.subplot(1, 10, i+1)
        plt.imshow((predictions[i, :, :] + 1)/2)
        plt.axis('off')

#    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch, label_batch in dataset:
            train_step(image_batch, label_batch)
        if epoch%10 == 0:
            generate_images(gen, noise_seed, cat_seed, epoch)
    generate_images(gen, noise_seed, cat_seed, epoch)
EPOCHS = 200
train(dataset, EPOCHS)