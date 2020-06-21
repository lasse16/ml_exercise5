class y_axis_classifier():
    def __init__(self, step):
        self.step = step

    def classify(self,point):
        return (int(point[1] > self.step) * 2) - 1

    def __str__(self):
        return "y-axis at {}".format(self.step)

class x_axis_classifier():
    def __init__(self, step):
        self.step = step

    def classify(self, point):
        return (int(point[0] > self.step) * 2) - 1

    def __str__(self):
        return "x-axis at {}".format(self.step)

