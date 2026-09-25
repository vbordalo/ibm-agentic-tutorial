import gradio as gr

from run_service import run_workflow

# Gradio user interface

with gr.Blocks() as demo:
    gr.Markdown("""
        # 🤖 Data-Science Assistant (LangGraph + local model)
        1️⃣ Add one or more files to the workspace.  
        2️⃣ Describe the analysis or task you want in plain English.  
        3️⃣ Press Run - the assistant will inspect the workspace, plan, code, execute, review, and return the result.
    """)

    with gr.Row():
        file_input = gr.File(label="📂 Workspace files (optional)",
                             file_types=[".csv", ".xlsx", ".xls", ".data"],
                             file_count="multiple",
                             )
        task_input = gr.Textbox(label="📝 Task description",
                                placeholder="e.g. train a linear regression model",
                                lines=3)

    run_btn = gr.Button("🚀 Run", variant="primary")

    planner_box = gr.Textbox(label="🗒️ Planner - numbered steps", lines=6)
    coder_box   = gr.Code(label="👩‍💻 Coder - generated Python code", language="python", lines=12)
    executor_out = gr.Textbox(label="⚙️ Executor - stdout", lines=6)
    executor_err = gr.Textbox(label="❌ Executor - error (if any)", lines=6)
    reviewer_box = gr.Textbox(label="🧐 Reviewer - suggestion (if any)", lines=6)
    final_box    = gr.Textbox(label="🎉 Final output", lines=8)

    def on_click(task, uploaded_files):
        out = run_workflow(task, uploaded_files)
        return (
            out["planner"], out["coder"], out["executor_output"],
            out["executor_error"], out["reviewer"], out["final_output"],
        )

    run_btn.click(fn=on_click, inputs=[task_input, file_input], outputs=[planner_box, coder_box, executor_out, executor_err, reviewer_box, final_box])

    
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Default())