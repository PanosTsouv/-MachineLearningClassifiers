class NaiveBaseImp:

    def __init__(self, label_cls = [0, 1], attribute_cls = [0, 1]) -> None:
        self.true_pos = 0;
        self.false_pos = 0;
        self.true_neg = 0;
        self.false_neg = 0;
        self.label_cls = label_cls
        self.attribute_cls = attribute_cls

    def train(self, train_attributes_samples, train_labels):
        pass

    def test(self, test_attributes_samples, test_labels):
        pass

    def _classify():
        pass