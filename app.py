import gradio as gr
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
try:
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    
    with open("logistic_regression.pkl", "rb") as lr_file:
        logistic_regression = pickle.load(lr_file)
    
    with open("perceptron.pkl", "rb") as perceptron_file:
        perceptron = pickle.load(perceptron_file)

except FileNotFoundError:
    print("Training models...")
    df = pd.read_excel("Student-Employability-Datasets.xlsx", sheet_name="Data")
    X = df.iloc[:, 1:-2].values
    y = (df["CLASS"] == "Employable").astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    logistic_regression = LogisticRegression(random_state=42)
    logistic_regression.fit(X_train_scaled, y_train)
    perceptron = Perceptron(random_state=42)
    perceptron.fit(X_train_scaled, y_train)
    with open("scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    with open("logistic_regression.pkl", "wb") as lr_file:
        pickle.dump(logistic_regression, lr_file)
    with open("perceptron.pkl", "wb") as perceptron_file:
        pickle.dump(perceptron, perceptron_file)
def predict_employability(name, ga, mos, pc, ma, sc, api, cs):
    if not name:
        name = "The candidate"
    
    input_data = np.array([[ga, mos, pc, ma, sc, api, cs]])
    input_scaled = scaler.transform(input_data)
    pred_lr = logistic_regression.predict(input_scaled)[0]
    pred_perceptron = perceptron.predict(input_scaled)[0]
    prediction = 1 if (pred_lr + pred_perceptron) >= 1 else 0
    if prediction == 1:
        return f"Congratulations!  {name} is Employable 😊🎉🎉"
    else:
        return f"{name}: Work Hard and Improve yourself! 💪"
def clear_inputs():
    return "", None, None, None, None, None, None, None, ""  # Reset sliders to None and clear textboxes
with gr.Blocks() as app:
    gr.Markdown("# 🎓 Employability Evaluation 🚀")
    
    gr.Markdown(
        "### 🏆 Find Out Your Employability Score! \n"
        "Use the sliders below to rate yourself on key employability skills, with 1 being 'Needs Improvement' and 5 being 'Excellent'"
    )

    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Name")
            ga = gr.Slider(1, 5, step=1, label="General Appearance")
            mos = gr.Slider(1, 5, step=1, label="Manner of Speaking")
            pc = gr.Slider(1, 5, step=1, label="Physical Condition")
            ma = gr.Slider(1, 5, step=1, label="Mental Alertness")
            sc = gr.Slider(1, 5, step=1, label="Self Confidence")
            api = gr.Slider(1, 5, step=1, label="Ability to Present Ideas")
            cs = gr.Slider(1, 5, step=1, label="Communication Skills")
            
            with gr.Row():
                predict_btn = gr.Button("Get Yourself Evaluated 🎯")
                clear_btn = gr.Button("Clear 🔄")

        with gr.Column():
            result_output = gr.Textbox(label="Employability Prediction")
    predict_btn.click(
        fn=predict_employability,
        inputs=[name, ga, mos, pc, ma, sc, api, cs],
        outputs=[result_output],  
    )

    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[name, ga, mos, pc, ma, sc, api, cs, result_output]
    )
app.launch(share=True)
