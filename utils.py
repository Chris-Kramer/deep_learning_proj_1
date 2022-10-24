

##############################
# Part one - Data generators #
##############################
def generate_data(targ_size: tuple[int, int], batch_size: int) -> ImageDataGenerator:
    """
    Creates datagenerators with the specified target dimensions and batchsizes
    """
    # ------ Training data ------
    train_datagen = ImageDataGenerator(
        rescale=1./255,) # Scale between 0 and 1
        #rotation_range=40, # Random rotations of 40 degree 
        #width_shift_range=0.2, # Randomly Shift pixels between 0% and 20% in width
        #height_shift_range=0.2, # Randomly shift pixels between 0% and 20% in height
        #shear_range=0.2, # Shear angle in counter-clockwise direction in degrees
        #zoom_range=0.2, # Zoom in between 0 and 20 %
        #horizontal_flip=True,
        #vertical_flip = True)

    train_generator = train_datagen.flow_from_directory(
        directory = r"catdog_data/train",
        target_size = targ_size,
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode='binary',
        shuffle = True) # Hypotese: Giver mere generaliserbar model, hvis shuffle er true

    # ------ Test data ------
    test_val_datagen = ImageDataGenerator(rescale=1./255) # Used for test and val, since we do not want to augment them
    test_generator = test_val_datagen.flow_from_directory(
        directory="catdog_data/test",
        target_size = targ_size, # Optimal solution = 224x224 eller 112x112
        color_mode="rgb",
        batch_size = batch_size,
        class_mode= "binary",
        shuffle = False,   
    )

    # ------ Validation data ------
    val_generator = test_val_datagen.flow_from_directory( 
        directory = "catdog_data/validation",
        target_size = targ_size,
        color_mode = "rgb",
        batch_size= batch_size,
        class_mode='binary')

    # Full training data set
    full_train_generator = train_datagen.flow_from_directory(
        directory = r"catdog_data/train_val",
        target_size = targ_size,
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode='binary',
        shuffle = True)

    return train_generator, val_generator, test_generator, full_train_generator
