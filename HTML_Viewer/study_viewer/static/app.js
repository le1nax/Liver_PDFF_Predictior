let authToken = null;

const authStatus = document.getElementById('auth-status');
const loginBtn = document.getElementById('login');
const passwordInput = document.getElementById('password');
const authSection = document.querySelector('.auth');
const content = document.getElementById('content');
const totalEl = document.getElementById('total');
const loadedEl = document.getElementById('loaded');
const casesEl = document.getElementById('cases');

async function login() {
  const password = passwordInput.value.trim();
  if (!password) {
    authStatus.textContent = 'Enter the password to continue.';
    return;
  }

  authStatus.textContent = 'Checking password...';
  const res = await fetch('/api/auth', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password })
  });

  if (!res.ok) {
    authStatus.textContent = 'Invalid password.';
    return;
  }

  const data = await res.json();
  authToken = data.token;
  authStatus.textContent = 'Access granted.';
  if (authSection) {
    authSection.classList.add('hidden');
  }
  await loadStudy();
}

async function loadStudy() {
  const res = await fetch('/api/study', {
    headers: { Authorization: `Bearer ${authToken}` }
  });

  if (!res.ok) {
    authStatus.textContent = 'Session expired. Please login again.';
    content.classList.add('hidden');
    return;
  }

  const data = await res.json();
  totalEl.textContent = data.total_cases;
  loadedEl.textContent = data.cases.length;

  casesEl.innerHTML = '';
  data.cases.forEach((id) => {
    const div = document.createElement('div');
    div.className = 'case';
    div.textContent = id;
    casesEl.appendChild(div);
  });

  content.classList.remove('hidden');
}

loginBtn.addEventListener('click', login);
passwordInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    login();
  }
});
