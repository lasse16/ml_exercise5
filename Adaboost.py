import math


class Adaboost():

    def __init__(self, data_points, weak_classifiers):
        self.importance = {}
        self.weak_classifiers = weak_classifiers
        self.best_classifiers = []
        self.data_points = [tuple(point) for point in data_points]

        uniform_importance = 1 / len(data_points)

        for point in self.data_points:
            self.importance[point] = uniform_importance

    def one_step(self):

        max_error = 0
        corresponding_classifier = 0

        for classifier in self.weak_classifiers:
            error = self.compute_error(classifier)
            if error > max_error:
                max_error = error
                corresponding_classifier = classifier

        confidence = self.calculate_confidence(max_error)

        self.best_classifiers += [(confidence, corresponding_classifier)]

        self.update_importance(corresponding_classifier, confidence)

    def train(self, steps):
        for i in range(steps):
            self.one_step()

    def classify(self, point):
        total = 0
        for confidence, classifier in self.best_classifiers:
            total += confidence * classifier(point)

        return math.copysign(1, total)

    def calculate_confidence(self, error):
        confidence = 0.5 * math.log((1 - error) / error)
        return confidence

    def compute_error(self, classifier):
        total = 0

        for point in self.data_points:
            label = point[2]
            classified = classifier(point)
            error = 1
            if label == classified:
                error = 0

            total += error * self.importance[point]

        return total

    def update_importance(self, classifier, confidence):
        normalizer = 0
        for point in self.data_points:
            label = point[2]
            new_importance = math.pow(self.importance[point], -confidence * label * classifier(point))
            self.importance[point] = new_importance
            normalizer += new_importance

        for point, importance in self.importance.items():
            self.importance[point] = (1 / normalizer) * importance
