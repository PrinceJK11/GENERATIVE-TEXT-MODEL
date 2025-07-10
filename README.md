# GENERATIVE-TEXT-MODEL

COMPANY: CODTECH IT SOLUTIONS

NAME: SOURAV PAL CHAUDHURI

ITERN ID: CT08DL242

DOMAIN: AI

DURATION: 8 WEEKS

MENTOR: NEELA SANTOSH

## 🚀 Features

- ✅ **GPT-2 integration** with automated fallback to LSTM or templates.
- ✅ **Custom LSTM model** with improved architecture (dropout, batch normalization, multiple layers).
- ✅ **Template-based fallback generator** for consistent outputs even without pre-trained models.
- ✅ **Training on high-quality, topic-specific text data**.
- ✅ **Interactive CLI mode** for generating custom text on the fly.
- ✅ **Graphical loss visualization** using `matplotlib`.
- ✅ No external training files needed – training data is built-in!

---

## 🧠 Model Architectures

### GPT-2 (via Hugging Face Transformers)
- Loaded dynamically with fallback if unavailable
- Used for high-quality paragraph generation

### Enhanced LSTM Model
- Embedding layer: 256 dimensions  
- Hidden layer: 512 units  
- 3 LSTM layers with dropout and batch normalization  
- Trained on a custom vocabulary generated from topic-based text samples  

---

## 📁 Project Structure

.
├── enhanced_text_generator.py   # Main script (the code above)
├── README.md                    # This file
🧪 Demonstration
Training Progress Example
yaml
Epoch 5/30, Loss: 3.0152
Epoch 10/30, Loss: 2.6231
Epoch 15/30, Loss: 2.2915
...
Sample Prompt Output
Prompt: Artificial intelligence is transforming

🤖 LSTM Output:

vbnet
Artificial intelligence is transforming industries and businesses around the world by offering smarter, faster, and more efficient ways to solve complex problems.
📝 GPT-2/Template Output:

rust
Artificial intelligence is transforming unprecedented opportunities for innovation and growth. The advancement of technology has led to The potential for further innovation remains limitless.
🛠️ Requirements
Make sure you have Python 3.7+ and the following libraries installed:

bash
pip install torch transformers matplotlib seaborn pandas numpy
If you face any issues with installing transformers or torch, make sure your Python version and OS are compatible.

▶️ How to Run
Just execute the main script to launch training and enter the interactive mode.

bash
python enhanced_text_generator.py
You’ll be able to:

View training progress

Generate sample outputs from pre-defined prompts

Enter your own prompt for dynamic generation

Exit with quit or Ctrl+C

📌 Topics Covered
Technology

Science

Business

Health

Each topic contains multiple high-quality training paragraphs to enhance contextual coherence during generation.

📊 Visualization
After training, loss graphs are displayed using matplotlib:

Overall Loss

Zoomed Loss (Last 20 Epochs)

