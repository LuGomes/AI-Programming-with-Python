# PURPOSE: Classifies pet images using a pretrained CNN model, compares these
#          classifications to the true identity of the pets in the images, and
#          summarizes how well the CNN performed on the image classification task.
#          Note that the true identity of the pet (or object) in the image is
#          indicated by the filename of the image. Therefore, your program must
#          first extract the pet image label from the filename before
#          classifying the images using the pretrained CNN model. With this
#          program we will be comparing the performance of 3 different CNN model
#          architectures to determine which provides the 'best' classification.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
from time import time

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Imports functions created for this program
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels
from classify_images import classify_images
from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats
from print_results import print_results

# Main program function defined below


def main():
    # Measures total program runtime by collecting start time
    start_time = time()

    # This function retrieves 3 Command Line Arguments as input from
    # the user running the program from a terminal window.
    in_arg = get_input_args()
    check_command_line_arguments(in_arg)

    # Define get_pet_labels function within the file get_pet_labels.py
    results = get_pet_labels(in_arg.dir)
    check_creating_pet_image_labels(results)

    # Creates Classifier Labels with classifier function, Compares Labels,
    # and adds these results to the results dictionary - results
    classify_images(in_arg.dir, results, in_arg.arch)
    check_classifying_images(results)

    # Adjusts the results dictionary to determine if classifier correctly
    # classified images as 'a dog' or 'not a dog'. This demonstrates if
    # model can correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(results, in_arg.dogfile)
    check_classifying_labels_as_dogs(results)

    # Creates the results statistics dictionary that contains counts & percentages.
    results_stats = calculates_results_stats(results)
    check_calculating_results(results, results_stats)

    # Prints summary results, incorrect classifications of dogs (if requested)
    # and incorrectly classified breeds (if requested)
    print_results(results, results_stats, in_arg.arch, True, True)

    end_time = time()

    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time % 3600)/60))+":"
          + str(round((tot_time % 3600) % 60)))
    final_results = """
    *** ----------------------------------Project Results -------------------------------------------- ***
    Given our results, the "best" model architecture is VGG.
    It out-performed both of the other architectures when considering both objectives 1 and 2.
    ResNet did classify dog breeds better than AlexNet,
    but only VGG and AlexNet were able to classify "dogs" and "not-a-dog"with 100% accuracy.
    The model VGG was the one that was able to classify "dogs" and "not-a-dog" with 100% accuracy and
    had the best performance regarding breed classification with 93.3% accuracy.
    If short runtime is of essence, one could argue that ResNet is preferrable to VGG.
    The resulting loss in accuracy in the dog breed classification is only 3.6% (VGG: 93.3% and ResNet: 89.7%).
    There is also some loss in accuracy predicting not-a-dog images of 9.1% (VGG: 100% and ResNet: 90.9%).
    *** ---------------------------------------------------------------------------------------------- ***
    """
    print(final_results)


# Call to main function to run the program
if __name__ == "__main__":
    main()
