// Samples for quick testing
const SAMPLES = {
    dialogue: `Amanda: I baked cookies for our team meeting today!
Jerry: Wow Amanda, you're the best! What flavor?
Amanda: Double chocolate chip with sea salt.
Jerry: Save two for me please, I am running slightly late from the client call.
Amanda: No problem, I kept a separate box in the breakroom for you.
Jerry: Thanks a lot!`,
    review: `I recently purchased a new smartphone, and I'm really impressed with its features. The camera quality is amazing, and the battery life lasts for almost two days on a single charge. The processor is also very fast, allowing me to multitask without any lag. Overall, it's a great device for the price.`,
    meeting: `Sarah: Let's review the timeline for the Q3 release.
David: Development is 90% done, but QA reported two blockers in authentication.
Elena: The backend team is deploying a hotfix this afternoon to unblock QA.
Sarah: Perfect. If tests pass tomorrow, we can schedule deployment for Thursday 10 AM.
David: Agreed. I will notify the stakeholders.`
};

document.addEventListener('DOMContentLoaded', () => {
    const inputText = document.getElementById('inputText');
    const summarizeBtn = document.getElementById('summarizeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const copyBtn = document.getElementById('copyBtn');
    const charCount = document.getElementById('charCount');
    const wordCount = document.getElementById('wordCount');
    const alertBox = document.getElementById('alertBox');
    const alertMessage = document.getElementById('alertMessage');
    const emptyState = document.getElementById('emptyState');
    const loadingState = document.getElementById('loadingState');
    const summaryContent = document.getElementById('summaryContent');
    const summaryText = document.getElementById('summaryText');
    const originalWordsEl = document.getElementById('originalWords');
    const summaryWordsEl = document.getElementById('summaryWords');
    const reductionRateEl = document.getElementById('reductionRate');
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toastMessage');

    // Update character & word counters
    function updateCounters() {
        const text = inputText.value;
        charCount.textContent = text.length.toLocaleString();
        
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        wordCount.textContent = words.toLocaleString();

        // Enable/Disable summarize button based on content
        summarizeBtn.disabled = !text.trim();
    }

    inputText.addEventListener('input', updateCounters);

    // Load sample text
    window.loadSample = function(key) {
        if (SAMPLES[key]) {
            inputText.value = SAMPLES[key];
            updateCounters();
            hideAlert();
            inputText.focus();
        }
    };

    // Clear input
    clearBtn.addEventListener('click', () => {
        inputText.value = '';
        updateCounters();
        hideAlert();
        resetSummaryView();
        inputText.focus();
    });

    // Reset summary view
    function resetSummaryView() {
        emptyState.style.display = 'flex';
        loadingState.style.display = 'none';
        summaryContent.style.display = 'none';
        summaryText.textContent = '';
        copyBtn.style.display = 'none';
    }

    // Show Alert
    function showAlert(msg) {
        alertMessage.textContent = msg;
        alertBox.style.display = 'flex';
    }

    // Hide Alert
    function hideAlert() {
        alertBox.style.display = 'none';
    }

    // Show Toast Notification
    function showToast(msg) {
        toastMessage.textContent = msg;
        toast.classList.add('show');
        setTimeout(() => {
            toast.classList.remove('show');
        }, 2500);
    }

    // Copy Summary to Clipboard
    copyBtn.addEventListener('click', async () => {
        const textToCopy = summaryText.textContent;
        if (!textToCopy) return;

        try {
            await navigator.clipboard.writeText(textToCopy);
            showToast('Summary copied to clipboard!');
        } catch (err) {
            // Fallback
            const textarea = document.createElement('textarea');
            textarea.value = textToCopy;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
            showToast('Summary copied to clipboard!');
        }
    });

    // Handle Keyboard Shortcut Ctrl+Enter / Cmd+Enter
    inputText.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            if (!summarizeBtn.disabled) {
                generateSummary();
            }
        }
    });

    // Summarize Button Click
    summarizeBtn.addEventListener('click', generateSummary);

    // Main Summarization Function
    async function generateSummary() {
        const rawText = inputText.value.trim();
        if (!rawText) {
            showAlert('Please enter some text or dialogue to summarize.');
            return;
        }

        hideAlert();
        emptyState.style.display = 'none';
        summaryContent.style.display = 'none';
        loadingState.style.display = 'flex';
        summarizeBtn.disabled = true;
        copyBtn.style.display = 'none';

        const startTime = performance.now();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ text: rawText })
            });

            const data = await response.json();

            if (!response.ok) {
                const errorDetail = data.detail || data.error || 'Failed to generate summary.';
                throw new Error(errorDetail);
            }

            const summary = data.summary || '';
            const origWordCount = rawText.split(/\s+/).length;
            const sumWordCount = summary.trim() ? summary.trim().split(/\s+/).length : 0;
            const reduction = origWordCount > 0 
                ? Math.max(0, Math.round(((origWordCount - sumWordCount) / origWordCount) * 100))
                : 0;

            summaryText.textContent = summary;
            originalWordsEl.textContent = origWordCount;
            summaryWordsEl.textContent = sumWordCount;
            reductionRateEl.textContent = `${reduction}%`;

            loadingState.style.display = 'none';
            summaryContent.style.display = 'flex';
            copyBtn.style.display = 'inline-flex';

        } catch (error) {
            console.error('Summarization error:', error);
            loadingState.style.display = 'none';
            emptyState.style.display = 'flex';
            showAlert(`Error: ${error.message || 'Something went wrong while connecting to the backend.'}`);
        } finally {
            summarizeBtn.disabled = false;
        }
    }

    // Initial counter setup
    updateCounters();
});
