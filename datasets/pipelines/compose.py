class Compose:
    """
    Compose a sequence of transforms.

    Each transform takes a `results` dictionary,
    modifies it, and returns it.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, results):
        for transform in self.transforms:
            results = transform(results)
        return results

