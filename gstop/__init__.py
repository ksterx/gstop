from .stopper import GenerationStopper

__all__ = ["GenerationStopper", "STOP_TOKENS_REGISTERY"]

STOP_TOKENS_REGISTRY = {
    "mistral": {
        "###": [[774], [27332]],
        "\n\n": [13, 13],
        "\n相手:": [13, 29367, 29427, 28747],
        "\nあなた:": [13, 29674, 29270, 29227, 28747],
        "\nUser:": [13, 730, 28747],
        "\nAssistant:": [13, 7226, 11143, 28747],
        "\nYou:": [13, 1976, 28747],
        "\nFollower:": [13, 28765, 793, 1072, 28747],
    }
}
