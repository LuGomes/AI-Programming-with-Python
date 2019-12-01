# PURPOSE: Create a function print_results that prints the results statistics
#          from the results statistics dictionary (results_stats_dic). It
#          should also allow the user to be able to print out cases of misclassified
#          dogs and cases of misclassified breeds of dog using the Results
#          dictionary (results_dic).


def print_results(results_dic, results_stats_dic, model,
                  print_incorrect_dogs=False, print_incorrect_breed=False):
    """
    Prints summary results on the classification and then prints incorrectly
    classified dogs and incorrectly classified dog breeds if user indicates
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifier labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
      results_stats_dic - Dictionary that contains the results statistics (either
                   a  percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
      model - Indicates which CNN model architecture will be used by the
              classifier function to classify the pet images,
              values must be either: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and
                             False doesn't print anything(default) (bool)
      print_incorrect_breed - True prints incorrectly classified dog breeds and
                              False doesn't print anything(default) (bool)
    Returns:
           None - simply printing results.
    """
    print('\n\n*** Results Summary for CNN Model Architecture: {}'.format(model.upper()))
    print('N Images: {}'.format(results_stats_dic['n_images']))
    print('N Dog Images: {}'.format(results_stats_dic['n_dogs_img']))
    print('N Not-Dog Images: {}'.format(results_stats_dic['n_notdogs_img']))

    for key, percentage in results_stats_dic.items():
        if 'pct' in key:
            print(f'{key}: {percentage}')

    if print_incorrect_dogs and \
            (results_stats_dic['n_correct_dogs'] + results_stats_dic['n_correct_notdogs'] != results_stats_dic['n_images']):
        print("\nINCORRECT Dog/NOT Dog Assignments:")
        for classification_list in results_dic.values():
            if sum(classification_list[3:]) == 1:
                print("Real is a dog: {}   Classifier is a dog: {}".format(bool(classification_list[3]),
                                                                           bool(classification_list[4])))

    if print_incorrect_breed and \
            results_stats_dic['n_correct_dogs'] != results_stats_dic['n_correct_breed']:
        print('\nINCORRECT Dog Breed Assignment:')
        for classification_list in results_dic.values():
            if (classification_list[2] == 0 and sum(classification_list[3:]) == 2):
                print('Pet image label: {}   Classifier label: {}'.format(
                    classification_list[0], classification_list[1]))
