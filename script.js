const messageInput = document.getElementById('messageInput');
const checkBtn = document.getElementById('checkBtn');
const clearBtn = document.getElementById('clearBtn');
const result = document.getElementById('result');
const resultText = document.getElementById('resultText');

function updateResult(type, text) {
  result.className = `result ${type}`;
  resultText.textContent = text;
}

function setLoading() {
  updateResult('loading', 'Checking...');
}

async function checkMessage() {
  const message = messageInput.value.trim();

  if (!message) {
    updateResult('neutral', 'Please enter a message first.');
    messageInput.focus();
    return;
  }

  setLoading();

  try {
    const response = await fetch('http://127.0.0.1:8000/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Prediction failed');
    }

    if (data.is_spam) {
      updateResult('spam', 'Spam message detected');
    } else {
      updateResult('safe', 'Not Spam');
    }
  } catch (error) {
    updateResult('neutral', 'Unable to reach the AI server.');
    console.error(error);
  }
}

checkBtn.addEventListener('click', checkMessage);
clearBtn.addEventListener('click', () => {
  messageInput.value = '';
  updateResult('neutral', 'Waiting for a message...');
  messageInput.focus();
});

messageInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
    checkMessage();
  }
});
