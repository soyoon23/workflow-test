"""Visualize the Plan-Act workflow graph."""

from src.llm.client import LLMClient
from src.prompts.registry import PromptRegistry
from src.skills.registry import SkillRegistry
from src.tools.registry import ToolRegistry
from src.workflow.components import WorkflowComponents
from src.workflow.graph import create_workflow


def main():
    # Initialize components
    components = WorkflowComponents(
        llm=LLMClient(),
        prompts=PromptRegistry(),
        tools=ToolRegistry(),
        skills=SkillRegistry(),
    )

    # Create workflow
    print("=== Workflow Graph ===")
    workflow = create_workflow(components)

    # Get the graph
    graph = workflow.get_graph()

    # Print Mermaid diagram (text format)
    print("=== Mermaid Diagram ===")
    print(graph.draw_mermaid())
    print()

    # Save as PNG (requires graphviz or pygraphviz)
    try:
        png_data = graph.draw_mermaid_png()
        with open("workflow_graph.png", "wb") as f:
            f.write(png_data)
        print("Graph saved to: workflow_graph.png")
    except Exception as e:
        print(f"Could not save PNG (install graphviz): {e}")
        print("\nYou can copy the Mermaid diagram above and paste it at:")
        print("https://mermaid.live/")


if __name__ == "__main__":
    main()
