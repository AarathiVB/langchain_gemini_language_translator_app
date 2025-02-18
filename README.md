# **Google Gemini-1.5-Pro Language Translator App**

## **Project Overview**
This project is a **language translation application** built using **Google Gemini-1.5-Pro**, **LangChain**, and **Streamlit**. The application translates input text from **English to German** using **LangChain's prompt chaining** and **Google's Gemini LLM**.

---

## **Features**
- Uses **Google Gemini-1.5-Pro** for accurate translations.
- Implements **LangChain's ChatPromptTemplate** for structured prompts.
- Uses **Streamlit** for a simple and interactive UI.
- Supports **LangChain's implicit chaining** for efficient LLM invocation.
- Ensures clean text output using **StrOutputParser**.

---

## **Prerequisites**
Ensure you have the following installed:
- **Python 3.8+**
- **Google API Key** (for accessing Gemini)
- **Streamlit**
- **LangChain & Dependencies**

---

## **Environment Setup:**
* py -3.10 -m venv myvenv
    
* myvenv\Scripts\activate

* python -m pip install --upgrade pip
* pip install --upgrade --quiet  langchain-google-genai pillow
* pip install streamlit
* pip install python-dotenv

* Get a Google API key: Head to https://ai.google.dev/gemini-api/docs/api-key to generate a Google AI API key.

---

## **Code Explanation**
### **1. Import Dependencies**
```python
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
```
- **Streamlit (`st`)** → Used to create the user interface.
- **LangChain (`ChatPromptTemplate`, `StrOutputParser`)** → Handles structured prompts and output parsing.
- **dotenv (`load_dotenv`)** → Loads API key securely from `.env`.

---

### **2. Load API Key**
```python
load_dotenv()
```
- Reads the environment variable for the **Google Gemini API Key**.

---

### **3. Define the Gemini LLM**
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)
```
- **Model:** `"gemini-1.5-pro"`
- **Temperature:** `0` (ensures consistent and deterministic translations)
- **Retries:** `2` (retries failed requests automatically)

---

### **4. Define the Prompt Template**
```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("human", "{input}")
    ]
)
```
- **System Message:** Instructs the model to act as a translator.
- **User Message:** Takes input dynamically for translation.

---

### **5. Define Output Parser**
```python
output_parser = StrOutputParser()
```
- Ensures clean and structured output.

---

### **6. Create the Processing Chain**
```python
chain = prompt | llm | output_parser
```
- **Uses LangChain's `|` operator** to chain the prompt → LLM → Output Parser.
- **Eliminates the need for `LLMChain`**, making the code more compact.

---

### **7. Streamlit UI**
```python
st.title('Langchain Language Translator with Gemini')
input_text = st.text_input("Write the sentence in English and it will be translated into German")
```
- Sets up a **user-friendly interface** with a title and input field.

---

### **8. Execute Translation**
```python
if input_text:
    st.write(chain.invoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": input_text
        }
    ))
```
- Passes the **input text**, source language (`English`), and target language (`German`) to Gemini.
- Displays the translated output.

---

## **Running the Application**
To launch the Streamlit app, run:
```bash
streamlit run gemini_applanguage_translator.py
```
This will open the **language translator** in your **web browser**.

---

## **References**
- [LangChain Documentation](https://python.langchain.com/)
- [Google Gemini API](https://ai.google.dev/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## **License**
This project is licensed under the **MIT License**.
