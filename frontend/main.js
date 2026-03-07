const questionInput = document.getElementById("question");
const sendBtn = document.getElementById("sendBtn");
const chatThreadNode = document.getElementById("chatThread");
const defaultButtonText = sendBtn.textContent;

const API_BASE_URL = (window.API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

function scrollChatToBottom() {
    chatThreadNode.scrollTop = chatThreadNode.scrollHeight;
}

function appendBubble(role, text, options = {}) {
    const bubble = document.createElement("article");
    bubble.classList.add("message", role === "user" ? "message-user" : "message-bot");

    if (options.isLoading) {
        bubble.classList.add("message-loading");
    }

    if (role !== "user") {
        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.textContent = "⚖️";
        bubble.appendChild(avatar);
    }

    const bubbleBody = document.createElement("div");
    bubbleBody.className = "bubble";
    if (options.isLoading) {
        bubbleBody.classList.add("bubble-loading");
    }

    const content = document.createElement("div");
    content.className = "bubble-content";
    content.textContent = text;

    bubbleBody.appendChild(content);
    bubble.appendChild(bubbleBody);

    chatThreadNode.appendChild(bubble);
    scrollChatToBottom();

    return { message: bubble, bubble: bubbleBody, content };
}

function setRequestState(isLoading) {
    sendBtn.disabled = isLoading;
    sendBtn.textContent = isLoading ? "Đang xử lý..." : defaultButtonText;
    questionInput.disabled = isLoading;
}

async function sendQuestion() {
    const question = questionInput.value.trim();
    if (question.length < 3) {
        appendBubble("bot", "Vui lòng nhập câu hỏi tối thiểu 3 ký tự.");
        return;
    }

    appendBubble("user", question);
    questionInput.value = "";

    const botMessage = appendBubble("bot", "Đang phân tích câu hỏi của bạn...", { isLoading: true });

    setRequestState(true);

    try {
        const response = await fetch(`${API_BASE_URL}/api/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                question,
                top_k: 5,
            }),
        });

        if (!response.ok) {
            const errorPayload = await response.json();
            throw new Error(errorPayload.detail || "Không thể gọi API");
        }

        const data = await response.json();
        botMessage.message.classList.remove("message-loading");
        botMessage.bubble.classList.remove("bubble-loading");
        botMessage.content.textContent = data.answer;
    } catch (error) {
        botMessage.message.classList.remove("message-loading");
        botMessage.bubble.classList.remove("bubble-loading");
        botMessage.content.textContent = error.message || "Có lỗi xảy ra";
    } finally {
        setRequestState(false);
        questionInput.focus();
        scrollChatToBottom();
    }
}

sendBtn.addEventListener("click", sendQuestion);

questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        event.preventDefault();
        sendQuestion();
    }
});
