from typing import Literal

# LangGraph
from langgraph.graph import END, START, StateGraph

from coder import coder_agent
from executor import executor_agent
from input_agent import ui_input_agent
from planner import planner_agent
from reviewer import reviewer_agent
from run_trace import RunTrace
from state import GraphState

# Workflow definition

MAX_ATTEMPTS = 5


def build_workflow(trace: RunTrace, llm) -> StateGraph:


    def planner_node(state):
        trace.increment("planner_calls")

        trace.log_event(
            event="node_started",
            node="planner",
            data={
                "task": state.get("task", ""),
                "dataset_info": state.get("dataset_info", ""),
            },
        )

        output = planner_agent(state, llm)

        trace.log_event(
            event="node_completed",
            node="planner",
            data={
                "instructions": output.get("instructions", ""),
            },
        )

        return output

    def coder_node(state):
        trace.increment("coder_calls")

        trace.log_event(
            event="node_started",
            node="coder",
            data={
                "instructions": state.get("instructions", ""),
                "dataset_info": state.get("dataset_info", ""),
            },
        )

        output = coder_agent(state, llm)

        trace.log_event(
            event="node_completed",
            node="coder",
            data={
                "code": output.get("code", ""),
            },
        )

        return output


    def executor_node(state):
        trace.increment("executor_calls")

        trace.log_event(
            event="node_started",
            node="executor",
            data={
                "attempt": state.get("attempts", 0) + 1,
                "code": state.get("code", ""),
            },
        )

        output = executor_agent(state)

        if output.get("exec_error"):
            trace.increment("execution_errors")

            trace.log_event(
                event="execution_failed",
                node="executor",
                data={
                    "attempt": output.get("attempts", 0),
                    "error": output.get("exec_error", ""),
                },
            )

        else:
            trace.log_event(
                event="execution_succeeded",
                node="executor",
                data={
                    "attempt": output.get("attempts", 0),
                    "output": output.get("exec_output", ""),
                },
            )

        return output


    def reviewer_node(state):
        trace.increment("reviewer_calls")

        trace.log_event(
            event="node_started",
            node="reviewer",
            data={
                "attempt": state.get("attempts", 0),
                "error": state.get("exec_error", ""),
                "code": state.get("code", ""),
            },
        )

        output = reviewer_agent(state, llm)

        if output.get("suggestions"):
            trace.log_event(
                event="reviewer_suggested_human",
                node="reviewer",
                data={
                    "suggestions": output.get("suggestions", ""),
                },
            )
        else:
            trace.log_event(
                event="reviewer_corrected_code",
                node="reviewer",
                data={
                    "code": output.get("code", ""),
                },
            )

        return output

    workflow = StateGraph(GraphState)

    workflow.add_node("input", ui_input_agent)
    workflow.add_node("planner", planner_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.add_edge(START, "input")
    workflow.add_edge("input", "planner")
    workflow.add_edge("planner", "coder")
    workflow.add_edge("coder", "executor")

    def after_executor(state: GraphState) -> Literal["reviewer", END]:
        if state["exec_error"] and state["attempts"] < MAX_ATTEMPTS:
            return "reviewer"
        return END

    def after_reviewer(state: GraphState) -> Literal["executor", END]:
        if state.get("suggestions"):
            return END
        return "executor"

    workflow.add_conditional_edges(
        "executor",
        after_executor,
        {
            "reviewer": "reviewer",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "reviewer",
        after_reviewer,
        {
            "executor": "executor",
            END: END,
        },
    )

    return workflow.compile()
