import _path_setup  # noqa: F401

from judge import HeuristicJudge, build_judge_prompt


if __name__ == "__main__":
    question = "What is RAG?"
    reference = "RAG retrieves relevant external context and gives it to an LLM before answer generation."
    answer = "RAG retrieves context for an LLM answer."

    print(build_judge_prompt(question, answer, reference))
    print(HeuristicJudge().score(question, answer, reference))
