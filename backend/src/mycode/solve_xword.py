import dotenv

from mycode.utils import argument_parsing
from mycode.utils.logger import logger
import dspy
from datasets import load_dataset

# Enable detailed tracebacks for debugging
import dspy.utils.parallelizer

dspy.utils.parallelizer.provide_traceback = True


def format_clue_with_length(clue: str, answer: str) -> str:
    """Format a crossword clue to include the answer length and word pattern."""
    # Create pattern with asterisks, preserving spaces
    pattern = "".join("*" if char != " " else " " for char in answer)
    total_letters = len(answer.replace(" ", ""))
    return f"{clue} ({total_letters} letters: {pattern})"


def load_crossword_examples(
    num_train: int,
    num_test: int,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    logger.info(
        f"Loading CrosswordQA dataset with {num_train} train and {num_test} test examples"
    )

    # Load the dataset
    dataset = load_dataset("albertxu/CrosswordQA")
    train_data = dataset["train"]
    test_data = dataset["validation"]

    for i in range(num_test):
        clue = format_clue_with_length(train_data[i]["clue"], train_data[i]["answer"])
        answer = train_data[i]["answer"]
        logger.info(f"Clue: {clue}. Answer: {answer}")

    # Convert to DSPy Example objects with length information
    train_examples = [
        dspy.Example(
            clue=format_clue_with_length(
                train_data[i]["clue"], train_data[i]["answer"]
            ),
            answer=train_data[i]["answer"],
        ).with_inputs("clue")
        for i in range(num_train)
    ]
    test_examples = [
        dspy.Example(
            clue=format_clue_with_length(test_data[i]["clue"], test_data[i]["answer"]),
            answer=test_data[i]["answer"],
        ).with_inputs("clue")
        for i in range(num_test)
    ]

    logger.info(
        f"Loaded {len(train_examples)} training examples and {len(test_examples)} test examples"
    )
    return train_examples, test_examples


def crossword_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
    """Metric function for evaluating crossword clue predictions."""
    predicted_answer = pred.answer.strip().lower()
    gold_answer = gold.answer.strip().lower()
    return predicted_answer == gold_answer


class CrosswordSolver(dspy.Module):
    """A DSPy module for solving crossword clues."""

    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict("clue -> answer")

    def forward(self, clue: str) -> dspy.Prediction:
        return self.generate_answer(clue=clue)


def main() -> None:
    logger.info("Starting Crossword Solver with DSPy")

    # Configure DSPy with language model
    lm = dspy.LM("openai/gpt-5-mini", temperature=1.0, max_tokens=16000)
    dspy.configure(lm=lm)

    # Load training and test data
    train_examples, test_examples = load_crossword_examples(num_train=50, num_test=50)

    crossword_solver = CrosswordSolver()
    optimizer = dspy.MIPROv2(
        metric=crossword_metric,
        auto="light",
    )

    logger.info("\n=== OPTIMIZING ===")
    optimized_solver = optimizer.compile(
        student=crossword_solver,
        trainset=train_examples,
    )

    logger.info("\n=== TESTING ===")
    logger.info(f"{lm.history}")


if __name__ == "__main__":
    dotenv.load_dotenv(".env")
    args = argument_parsing.parse_args(main)
    main(**args)
