FEATURE_EXTRACTORS = {}

def register_feature(name=None):
    def decorator(fn):
        FEATURE_EXTRACTORS[name or fn.__name__] = fn
        return fn
    return decorator