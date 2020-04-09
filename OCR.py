# The starting point of the backend works
# The image URL from UI is fed to the perform_ocr() here

import cv2
from Segmentation_words import get_words
from Segmentation_characters import get_characters
from User_input import get_string_from_nn

from Dict import correction


# input image should have white bg, black text
# raw_image
def perform_ocr(img_url):
    # use dictionary or not
    use_dict = True

    raw_image = cv2.imread(img_url, 0)

    # get all the words (as an numpy image array), words on each line, and maximum height on that line
    all_words, words_on_line, max_height_on_line = get_words(raw_image)
    print("No of lines =", len(words_on_line))
    print(words_on_line)
    print("No of words =", len(all_words))
    # to write the output into a file
    fp = open("output.txt", 'w')
    fp.truncate()
    count = 0
    for i in range(0, len(words_on_line)):
        for j in range(0, words_on_line[i]):
            all_characters = get_characters(all_words[count-1], max_height_on_line[i], i, j)
            if use_dict:
                print(correction(get_string_from_nn(all_characters))),
                fp.write(correction(get_string_from_nn(all_characters)))
                fp.write(" ")

            else:
                print(get_string_from_nn(all_characters))
                fp.write(get_string_from_nn(all_characters))
                fp.write(" ")

                # exit(0)
                # cv2.imshow("all_words[count]",all_words[count])
                # cv2.waitKey()
                count = count + 1

        print("\n")
        fp.write("\n")

    fp.close()
