# PURPOSE: Create a function classify_images that uses the classifier function
#          to create the classifier labels and then compares the classifier
#          labels to the pet image labels.

# Imports classifier function for using CNN to classify images
from classifier import classifier


def classify_images(images_dir, results_dic, model):
    """
    Creates classifier labels with classifier function, compares pet labels to
    the classifier labels, and adds the classifier label and the comparison of
    the labels to the results dictionary.

    Recall that dog names from the classifier function can be a string of dog
    names separated by commas when a particular breed of dog has multiple dog
    names associated with that breed. For example, you will find pet images of
    a 'dalmatian'(pet label) and it will match to the classifier label
    'dalmatian, coach dog, carriage dog' if the classifier function correctly
    classified the pet images of dalmatians.

    Parameters:
    images_dir - The (full) path to the folder of images that are to be
                  classified by the classifier function (string)
    results_dic - Results Dictionary with 'key' as image filename and 'value'
                  as a List. Where the list will contain the following items:
                index 0 = pet image label (string)
                index 1 = classifier label (string)
                index 2 = 1/0 (int)  where 1 = match between pet image
                  and classifier labels and 0 = no match between labels
    model - Indicates which CNN model architecture will be used by the
            classifier function to classify the pet images,
            values must be either: resnet alexnet vgg (string)
    Returns:
          None - results_dic is mutable data type so no return needed.
    """
    for image_filename, classification_list in results_dic.items():
        test_image = images_dir + image_filename
        classifier_label = classifier(test_image, model).lower().strip()
        pet_label = classification_list[0]

        classification_list.extend(
            [classifier_label, int(pet_label in classifier_label)])
