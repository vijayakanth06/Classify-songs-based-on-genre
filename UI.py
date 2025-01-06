import streamlit as st
from mod import predict_genre

def main():
    st.title("Music Genre Prediction")
    
    uploaded_file = st.file_uploader("Choose a WAV file", type="wav")
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Predict the genre
        genre = predict_genre("temp.wav")
        
        # Display the result
        st.write(f"The predicted genre is: {genre}")

if __name__ == "__main__":
    main()