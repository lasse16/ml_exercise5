from pprint import pprint
from numpy import arange
from matplotlib import pyplot as plt
from Adaboost import Adaboost
from AxisClassifier import y_axis_classifier, x_axis_classifier


def generate_x_axis_classifier(step):
    return x_axis_classifier(step)


def generate_y_axis_classifier(step):
    return y_axis_classifier(step)


def generate_classifiers():
    classifiers = []
    for step in arange(-10, 10, 0.1):
        classifiers += [generate_x_axis_classifier(step).classify, generate_y_axis_classifier(step).classify]

    return classifiers


def draw_classifier(classifier, plt):
    if(classifier is x_axis_classifier):
        plt.axvline(classifier.step)
    else:
        plt.axhline(classifier.step)

def main():
    filePath = r"C:\Users\Lasse\Desktop\ML_Exercise5\dataCircle.txt"
    dataPoints = []

    with open(filePath) as f:
        for line in f.readlines():
            line = line.strip()
            point = [float(x) for x in list(filter(lambda x: x != '', line.split(" ")))]
            dataPoints += [point]

    classifiers = generate_classifiers()

    booster = Adaboost(dataPoints, classifiers)

    booster.train(steps=10)

    total_correct_classified = 0
    for point in dataPoints:
        classified = booster.classify(point)
        if point[2] == classified:
            total_correct_classified += 1

    percent_correct_classified = total_correct_classified / len(dataPoints)

    positivePoints = list(filter(lambda x: x[2] == 1, dataPoints))
    negativePoints = list(filter(lambda x: x[2] == -1, dataPoints))

    for points, color in zip([positivePoints, negativePoints], ['green', 'red']):
        posX = [point[0] for point in points]
        posY = [point[1] for point in points]
        plt.scatter(posX, posY, c=color)

    for confidence, classifier in booster.best_classifiers:
        classifier = classifier.__self__
        draw_classifier(classifier, plt)
        print("confidence:{} - classifier{}".format(confidence,classifier))

    plt.show()

    print("total correct {} -> {} percent ".format(total_correct_classified, percent_correct_classified))


if __name__ == '__main__':
    main()
