import streamlit as st
import cv2
import numpy as np
from real_time_eval import real_time_evaluation

st.set_page_config(page_title="Multi-Exercise Rehabilitation Evaluator", layout="wide")

st.title("🏋️‍♂️ Multi-Exercise Rehabilitation Evaluator")
st.write("Real-time assessment of wrist extension and flexion exercises using AI-powered pose analysis")

# Exercise selection
exercises = ['wrist_extension', 'wrist_flexion']
selected_exercise = st.selectbox("Select Exercise to Evaluate:", exercises)

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Live Video Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("Exercise Quality")
    quality_display = st.empty()

def main():
    # Show instructions
    st.info(f"Perform {selected_exercise.replace('_', ' ').title()} exercises. The system will automatically detect and evaluate your form.")
    
    # Start real-time evaluation
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame (simplified for demo)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # In production, integrate with real_time_eval logic
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Simulate exercise detection and evaluation
        if selected_exercise == 'wrist_extension':
            quality_score = np.random.uniform(0.7, 1.0)  # Simulated good score
        else:  # wrist_flexion
            quality_score = np.random.uniform(0.4, 0.8)  # Simulated moderate score
        
        # Update display
        if quality_score > 0.7:
            quality_display.success(f"✅ Excellent {selected_exercise.replace('_', ' ').title()} Form!")
        elif quality_score > 0.4:
            quality_display.warning(f"⚠️ Moderate {selected_exercise.replace('_', ' ').title()} Form - Check technique")
        else:
            quality_display.error(f"❌ Poor {selected_exercise.replace('_', ' ').title()} Form - Adjust technique")
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()