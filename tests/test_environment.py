import importlib.util


def test_scientific_dependencies():
    required = [
        "numpy",
        "scipy",
        "sympy",
        "matplotlib",
        "networkx",
    ]

    for package in required:
        assert importlib.util.find_spec(package) is not None
