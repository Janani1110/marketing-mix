// static/app.js

// Universal Chart Configuration
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Outfit', sans-serif";
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 20, 35, 0.9)';
Chart.defaults.plugins.tooltip.padding = 12;
Chart.defaults.plugins.tooltip.cornerRadius = 8;
Chart.defaults.borderColor = 'rgba(255,255,255,0.05)';

// State
let charts = {};

// Global Navigation Handler
window.navigateTo = function(targetId) {
    // Show target section exclusively
    document.querySelectorAll('.view-section').forEach(sec => sec.classList.remove('active'));
    document.getElementById(targetId).classList.add('active');

    // Trigger loads automatically based on view
    if(targetId === 'time-series') loadTimeSeries();
    if(targetId === 'contributions') loadContributions();
    if(targetId === 'inspector') loadDataInspector();
};

// Format currency
const formatCurrency = (val) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

// 1. Time Series
async function loadTimeSeries() {
    try {
        const res = await fetch('/api/time-series');
        const data = await res.json();
        
        const labels = data.map(d => d.date.split(' ')[0]);
        const rev = data.map(d => d.revenue);
        const fb = data.map(d => d.spend_facebook);
        const gg = data.map(d => d.spend_google);
        const inf = data.map(d => d.spend_influencer);

        // Update Metrics
        const latestRev = rev[rev.length-1];
        const avg7 = rev.slice(-7).reduce((a,b)=>a+b,0)/7;
        const totalUnits = data.reduce((sum, d)=>sum+d.units_sold, 0);

        document.getElementById('ts-metrics').innerHTML = `
            <div class="metric-box">
                <div class="metric-label">Latest Revenue</div>
                <div class="metric-value">${formatCurrency(latestRev)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">7-Day Avg Revenue</div>
                <div class="metric-value">${formatCurrency(avg7)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total Units Sold</div>
                <div class="metric-value" style="color: #a855f7;">${new Intl.NumberFormat().format(totalUnits)}</div>
            </div>
        `;

        // Render Rev Chart
        if(charts.rev) charts.rev.destroy();
        charts.rev = new Chart(document.getElementById('revenueChart'), {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Revenue',
                    data: rev,
                    borderColor: '#38bdf8',
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: { responsive: true, interaction: { intersect: false, mode: 'index' } }
        });

        // Render Spend Chart (Area)
        if(charts.spend) charts.spend.destroy();
        charts.spend = new Chart(document.getElementById('spendChart'), {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'Google', data: gg, borderColor: '#f87171', backgroundColor: 'rgba(248, 113, 113, 0.2)', fill: true, tension: 0.4, pointRadius: 0 },
                    { label: 'Facebook', data: fb, borderColor: '#60a5fa', backgroundColor: 'rgba(96, 165, 250, 0.2)', fill: true, tension: 0.4, pointRadius: 0 },
                    { label: 'Influencer', data: inf, borderColor: '#c084fc', backgroundColor: 'rgba(192, 132, 252, 0.2)', fill: true, tension: 0.4, pointRadius: 0 }
                ]
            },
            options: { responsive: true, interaction: { intersect: false, mode: 'index' }, plugins: { filled: true } }
        });

    } catch(e) { console.error('Error loading time series:', e); }
}

// 2. Predictions
async function runPrediction() {
    const payload = {
        spend_google: document.getElementById('pred-google').value,
        spend_facebook: document.getElementById('pred-facebook').value,
        spend_influencer: document.getElementById('pred-influencer').value,
        discount: document.getElementById('pred-discount').value,
        promo_type: document.getElementById('pred-promo').value
    };

    document.getElementById('pred-result').innerText = 'Calculating...';
    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        document.getElementById('pred-result').innerText = formatCurrency(data.predicted_revenue);
    } catch(e) { document.getElementById('pred-result').innerText = 'Error'; }
}

// 3. Contributions
async function loadContributions() {
    try {
        const res = await fetch('/api/channel-contributions');
        const data = await res.json();
        const cont = data.contributions;

        if(charts.cont) charts.cont.destroy();
        charts.cont = new Chart(document.getElementById('contributionsChart'), {
            type: 'bar',
            data: {
                labels: ['Google', 'Facebook', 'Influencer'],
                datasets: [{
                    label: 'Δ Revenue',
                    data: [cont.spend_google, cont.spend_facebook, cont.spend_influencer],
                    backgroundColor: ['#f87171', '#60a5fa', '#c084fc'],
                    borderRadius: 6
                }]
            },
            options: { responsive: true, plugins: { legend: { display: false } } }
        });
    } catch(e) { console.error(e); }
}

// 4. Optimizer
async function runOptimization() {
    const payload = {
        budget: document.getElementById('opt-budget').value,
        step_size: document.getElementById('opt-step').value
    };

    document.getElementById('opt-metrics').innerHTML = `<p>Optimizing...</p>`;
    
    try {
        const res = await fetch('/api/optimize', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        const alloc = data.allocated_budget;

        document.getElementById('opt-metrics').innerHTML = `
            <div class="metric-box">
                <div class="metric-label">Initial Rev</div>
                <div class="metric-value" style="font-size: 1.8rem;">${formatCurrency(data.initial_prediction)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Optimized Rev</div>
                <div class="metric-value" style="font-size: 2.2rem; color: #38bdf8;">${formatCurrency(data.final_prediction)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Algorithm Iterations</div>
                <div class="metric-value" style="font-size: 1.5rem;">${data.iterations}</div>
            </div>
        `;

        if(charts.opt) charts.opt.destroy();
        charts.opt = new Chart(document.getElementById('optChart'), {
            type: 'doughnut',
            data: {
                labels: ['Google', 'Facebook', 'Influencer'],
                datasets: [{
                    data: [alloc.spend_google, alloc.spend_facebook, alloc.spend_influencer],
                    backgroundColor: ['#f87171', '#60a5fa', '#c084fc'],
                    borderWidth: 0,
                    hoverOffset: 4
                }]
            },
            options: { responsive: true, cutout: '70%', plugins: { legend: { position: 'right' } } }
        });
    } catch(e) { console.error(e); }
}

// 5. Retrain
async function runRetrain() {
    const btn = document.getElementById('retrain-btn');
    const logs = document.getElementById('retrain-logs');
    btn.innerText = 'Retraining Model...';
    logs.classList.add('hidden');

    try {
        const res = await fetch('/api/retrain', {method: 'POST'});
        const data = await res.json();
        
        btn.innerText = 'Trigger Full Retrain';
        logs.innerText = JSON.stringify(data[0].meta, null, 2);
        logs.classList.remove('hidden');
    } catch(e) {
        btn.innerText = 'Error! Try Again.';
    }
}

// 6. Data Inspector
async function loadDataInspector() {
    try {
        const res = await fetch('/api/data-inspector');
        const data = await res.json();
        
        if(!data.length) return;
        const headers = Object.keys(data[0]);
        
        document.getElementById('table-headers').innerHTML = headers.map(h => `<th>${h}</th>`).join('');
        document.getElementById('table-body').innerHTML = data.map(row => {
            return `<tr>${headers.map(h => `<td>${row[h]}</td>`).join('')}</tr>`;
        }).join('');
    } catch(e) { console.error(e); }
}
