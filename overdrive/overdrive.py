import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


class Overdriver:
    """
    Abstract base class for image transformations.
    """

    def __init__(self, probability, labeled):
        """
        :param probability: probability of applying this transformation to the image. Probability is manifested image-wise, not batch-wise.
        :param labeled: whether your processing labeled inputs (images, labels) or unlabaled input/
        """
        self.probability = tf.convert_to_tensor(probability)
        # in labeled=True, this object could be called on a batch of images.
        # in labaled=False, this object could be called on a (batch of images, batch of labels)
        self.call = self.transform_labeled if labeled else self.transform

    def __call__(self, *args, **kwargs):
        return self.call(*args)

    def __repr__(self):
        return str(self)

    def probably(self, transformed, images):
        """
        Randomly choosing between transformed and original images with the probability set while creating this transformation.

        :param transformed: transformed images.
        :param images: original images.
        :return: a batch of transformed and original images. If probability = 1., this would return a batch of all transformed images.
        """
        coins = tf.random.uniform(shape=images.shape[:1], minval=0.0, maxval=1.0)
        return tf.where(coins <= self.probability, transformed, images)

    def transform(self, images):
        """

        :param images: batch of images.
        :return: transformed batch of images.
        """
        pass

    def transform_labeled(self, images, labels):
        return (self.transform(images), labels)


class Tilter(Overdriver):
    """
    Vertical perspective transformation.

    Based of tensorflow-addons > image
    """

    def __init__(self, probability, size, maxdelta, batch_size, labeled=True):
        """

        :param probability: see overdrive.Overdriver base class.
        :param size: input image size. Note: int and not a tuple. For now, this supports square image inputs.
        :param maxdelta: float between 0. and 1.
        :param batch_size: batch size.
        :param labeled: see overdrive.Overdriver base class.
        """
        super(Tilter, self).__init__(probability, labeled)
        self.size = tf.convert_to_tensor(size)
        self.maxdelta = tf.convert_to_tensor(maxdelta)
        self.batch_size = batch_size
        self.ORIGINAL = tf.convert_to_tensor(
            [[[0, 0], [size, 0], [0, size], [size, size]]], dtype=tf.int32
        )
        self.ORIGINAL = tf.tile(self.ORIGINAL, [batch_size, 1, 1])
        self.TOP = tf.convert_to_tensor([[[-1, 0], [+1, 0], [0, 0], [0, 0]]], dtype=tf.int32)
        self.TOP = tf.tile(self.TOP, [batch_size, 1, 1])
        self.BOTTOM = tf.convert_to_tensor([[[0, 0], [0, 0], [-1, 0], [+1, 0]]], dtype=tf.int32)
        self.BOTTOM = tf.tile(self.BOTTOM, [batch_size, 1, 1])

    def transform(self, images):
        if images.shape[0] != self.batch_size:
            raise ValueError(
                "input batch size not equal to initially declared batch size. Please note that TFTilt require fixed batch size."
            )

        delta = tf.random.uniform(
            (self.batch_size, 1, 1),
            minval=0,
            maxval=tf.cast(self.maxdelta * tf.cast(self.size, tf.float32), tf.int32),
            dtype=tf.int32,
        )
        delta = tf.tile(delta, [1, 4, 2])

        top_bottom_mask = tf.cast(
            tf.random.uniform((self.batch_size, 1, 1), minval=0, maxval=1) > 0.5, tf.int32
        )
        top_bottom_mask = tf.tile(top_bottom_mask, [1, 4, 2])

        top_shift = self.TOP * top_bottom_mask
        bottom_shift = self.BOTTOM * (1 - top_bottom_mask)

        self.new_plane = self.ORIGINAL + delta * (top_shift + bottom_shift)

        eyes = tf.tile(tf.eye(2, batch_shape=(self.batch_size,), dtype=tf.int32), [1, 4, 1])
        matrix = tf.Variable(lambda: tf.zeros((self.batch_size, 8, 8), dtype=tf.int32))

        matrix = matrix[:, :, 2::3].assign(eyes)
        matrix = matrix[:, ::2, :2].assign(self.ORIGINAL)
        matrix = matrix[:, 1::2, 3:5].assign(self.ORIGINAL)
        matrix = matrix[:, ::2, -2:].assign(self.ORIGINAL)
        matrix = matrix[:, 1::2, -2:].assign(self.ORIGINAL)

        matrix = matrix[:, :, -2:].assign(
            matrix[:, :, -2:]
            * (-tf.tile(tf.reshape(self.new_plane, shape=(self.batch_size, 8, 1)), [1, 1, 2]))
        )

        matrix = tf.cast(matrix, dtype=tf.float32)

        vector = tf.reshape(self.new_plane, (self.batch_size, 8))
        vector = tf.cast(vector, dtype=tf.float32)

        MM = tf.matmul(
            tf.linalg.inv(tf.matmul(matrix, matrix, transpose_a=True)), matrix, transpose_b=True
        )

        coeffs = tf.linalg.matvec(MM, vector)

        coeffs = tf.reshape(coeffs, (self.batch_size, 8))

        return self.probably(tfa.image.transform(images, coeffs), images)

    def __str__(self):
        return "Overdrive.Titler(probability=%.2f, size=%d, maxdelta=%.2f, batch_size=%d)" % (
            self.probability,
            self.size,
            self.maxdelta,
            self.batch_size,
        )


class FreeRotator(Overdriver):
    """
    Rotation with the angle being anything between 0 and 360 degrees.

    Based of tensorflow-addons > image
    """

    def __init__(self, probability, degree, labeled=True):
        """

        :param probability: see overdrive.Overdriver base class.
        :param degree: maximum angle degree.
        :param labeled: see overdrive.Overdriver base class.
        """
        super(FreeRotator, self).__init__(probability, labeled)
        self.degree = degree
        self.radians = tf.convert_to_tensor(np.deg2rad(degree), dtype=tf.float32)

    def transform(self, images):
        angles = tf.random.uniform(
            shape=images.shape[:1], minval=-self.radians, maxval=self.radians
        )
        return self.probably(tfa.image.rotate(images, angles), images)

    def __str__(self):
        return "Overdrive.FreeRotator(probability=%.2f, degree=%d)" % (
            self.probability,
            self.degree,
        )


class NinetyRotator(Overdriver):
    """
    Rotation with an angle being randoming drawn between 90, 180, and 270.
    """

    def __init__(self, probability, labeled=True):
        super(NinetyRotator, self).__init__(probability, labeled)

    def transform(self, images):
        return self.probably(
            tf.image.rot90(images, k=tf.random.uniform((), minval=1, maxval=4, dtype=tf.int32)),
            images,
        )

    def __str__(self):
        return "Overdrive.NinetyRotator(probability=%.2f)" % (self.probability,)


class ChannelDisturber(Overdriver):
    """
    Randomly disturbs RGB channels to produce different white-balance.
    """

    def __init__(self, probability, factor, labeled=True):
        """

        :param probability: see overdrive.Overdriver base class.
        :param factor: float between 0. and .5, RGB channels will be multiplied by a number in (1-factor, 1+factor)
        :param labeled: see overdrive.Overdriver base class.
        """
        super(ChannelDisturber, self).__init__(probability, labeled)
        self.factor = factor

    def transform(self, images):
        disturbances = 1 + tf.random.uniform(
            shape=(images.shape[0], 1, 1, 3), minval=-self.factor, maxval=self.factor
        )

        return self.probably(images * disturbances, images)

    def __str__(self):
        return "Overdrive.ChannelDisturber(probability=%.2f, factor=%.2f)" % (
            self.probability,
            self.factor,
        )


class RandomSaturator(Overdriver):
    """
    Random shift in saturation.
    """

    def __init__(self, probability, maxfactor=1.25, minfactor=0.0, labeled=True):
        super(RandomSaturator, self).__init__(probability, labeled)
        self.maxfactor = maxfactor
        self.minfactor = minfactor

    def transform(self, images):
        return self.probably(
            tf.image.random_saturation(images, lower=self.minfactor, upper=self.maxfactor), images
        )

    def __str__(self):
        return "Overdrive.RandomSaturator(probability=%.2f, maxfactor=%.2f, minfactor=%.2f)" % (
            self.probability,
            self.maxfactor,
            self.minfactor,
        )


class ImageWiseStandardizer(Overdriver):
    """
    Wrapper of tf.image.per_image_standardization.
    """

    def __init__(self, labeled=True):
        super(ImageWiseStandardizer, self).__init__(1.0, labeled)

    def transform(self, images):
        return tf.image.per_image_standardization(images)

    def __str__(self):
        return "Overdrive.ImageWiseStandardizer()"


class RelativeCropper(Overdriver):
    """
    Random crop in range (MINAREA% - 100%) while preserving dimension.
    """

    def __init__(self, probability, minarea, target_size, labeled=True):
        super(RelativeCropper, self).__init__(probability, labeled)
        self.minarea = tf.convert_to_tensor(minarea, dtype=tf.float32)
        self.target_size = tf.convert_to_tensor(target_size, dtype=tf.int32)

    def transform(self, images):
        area = tf.random.uniform((), minval=self.minarea, maxval=1.0)
        intermediate_size = tf.cast(tf.cast(self.target_size, tf.float32) / self.minarea, tf.int32)

        return self.probably(
            tf.image.random_crop(
                tf.image.resize(images, size=intermediate_size),
                size=(images.shape[0], self.target_size[0], self.target_size[1], 3),
            ),
            images,
        )

    def __str__(self):
        return "Overdrive.RelativeCropper(probability=%.2f, minarea=%.2f, target_size=%s)" % (
            self.probability,
            self.minarea,
            str(self.target_size),
        )


class AbsoluteCropper(Overdriver):
    """
    Random crop given a target size.
    """

    def __init__(self, target_size, labeled=True):
        super(AbsoluteCropper, self).__init__(1.0, labeled)
        self.target_size = tf.convert_to_tensor(target_size, dtype=tf.int32)

    def transform(self, images):
        return tf.image.random_crop(
            images, size=(images.shape[0], self.target_size[0], self.target_size[1], 3)
        )

    def __str__(self):
        return "Overdrive.AbsoluteCropper(probability=%.2f,target_size=%s)" % (
            self.probability,
            str(self.target_size),
        )
