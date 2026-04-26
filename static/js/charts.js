const PALETTE = {
    accent: '#38bdf8',
    accent2: '#818cf8',
    accent3: '#34d399',
    danger: '#f87171',
    warn: '#fbbf24',
    muted: '#334155',
    text: '#e2e8f0',
};

const BASE_OPTIONS = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            labels: {
                color: PALETTE.text,
                font: { family: 'DM Mono', size: 11 },
                boxWidth: 10,
                padding: 16,
            },
        },
        tooltip: {
            backgroundColor: '#0e1420',
            borderColor: 'rgba(99,179,237,0.2)',
            borderWidth: 1,
            titleColor: PALETTE.accent,
            bodyColor: PALETTE.text,
            titleFont: { family: 'Syne', size: 12, weight: '700' },
            bodyFont: { family: 'DM Mono', size: 11 },
            padding: 12,
        },
    },
    scales: {
        x: {
            grid: { color: 'rgba(99,179,237,0.06)' },
            ticks: { color: PALETTE.muted, font: { family: 'DM Mono', size: 10 } },
        },
        y: {
            grid: { color: 'rgba(99,179,237,0.06)' },
            ticks: { color: PALETTE.muted, font: { family: 'DM Mono', size: 10 } },
        },
    },
};

function mergeDeep(target, source) {
    const out = Object.assign({}, target);
    for (const k in source) {
        if (source[k] && typeof source[k] === 'object' && !Array.isArray(source[k])) {
            out[k] = mergeDeep(target[k] || {}, source[k]);
        } else {
            out[k] = source[k];
        }
    }
    return out;
}

function opts(overrides = {}) {
    return mergeDeep(BASE_OPTIONS, overrides);
}

function makeChurn(ctx, data) {
    if (!data || !data.labels) return;
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: data.labels,
            datasets: [{
                data: data.values,
                backgroundColor: [PALETTE.danger, PALETTE.accent3],
                borderColor: '#080c14',
                borderWidth: 3,
                hoverOffset: 8,
            }],
        },
        options: opts({
            plugins: { legend: { position: 'bottom' } },
            scales: { x: { display: false }, y: { display: false } },
            cutout: '70%',
        }),
    });
}

function makeFeatureImportance(ctx, data) {
    if (!data || !data.labels) return;
    const colors = data.labels.map((_, i) =>
        `rgba(56, 189, 248, ${1 - i * 0.055})`
    );
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.labels,
            datasets: [{
                label: 'Importance',
                data: data.values,
                backgroundColor: colors,
                borderRadius: 4,
            }],
        },
        options: opts({
            indexAxis: 'y',
            plugins: { legend: { display: false } },
        }),
    });
}

function makeROC(ctx, data) {
    if (!data || !data.fpr) return;
    const pts = data.fpr.map((x, i) => ({ x, y: data.tpr[i] }));
    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                    label: 'ROC Curve',
                    data: pts,
                    showLine: true,
                    fill: false,
                    borderColor: PALETTE.accent,
                    pointRadius: 0,
                    borderWidth: 2,
                    tension: 0.2,
                },
                {
                    label: 'Random',
                    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                    showLine: true,
                    fill: false,
                    borderColor: PALETTE.muted,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    borderWidth: 1,
                },
            ],
        },
        options: opts({
            scales: {
                x: { min: 0, max: 1, title: { display: true, text: 'FPR', color: PALETTE.muted, font: { size: 10 } } },
                y: { min: 0, max: 1, title: { display: true, text: 'TPR', color: PALETTE.muted, font: { size: 10 } } },
            },
        }),
    });
}

function makeContractChurn(ctx, data) {
    if (!data || !data.labels) return;
    const datasets = Object.entries(data.datasets).map(([label, values], i) => ({
        label,
        data: values,
        backgroundColor: i === 0 ? PALETTE.accent3 : PALETTE.danger,
        borderRadius: 4,
    }));
    new Chart(ctx, {
        type: 'bar',
        data: { labels: data.labels, datasets },
        options: opts({ plugins: { legend: { position: 'bottom' } } }),
    });
}

function makeTenure(ctx, data) {
    if (!data || !data.labels) return;
    const datasets = Object.entries(data.datasets).map(([label, values], i) => ({
        label,
        data: values,
        backgroundColor: i === 0 ?
            'rgba(52,211,153,0.7)' : 'rgba(248,113,113,0.7)',
        borderRadius: 4,
    }));
    new Chart(ctx, {
        type: 'bar',
        data: { labels: data.labels, datasets },
        options: opts({
            plugins: { legend: { position: 'bottom' } },
            scales: { x: BASE_OPTIONS.scales.x, y: {...BASE_OPTIONS.scales.y, stacked: false } },
        }),
    });
}

function animateGauge(svgCircle, pct, color) {
    const r = parseFloat(svgCircle.getAttribute('r'));
    const circ = 2 * Math.PI * r;
    svgCircle.style.strokeDasharray = circ;
    svgCircle.style.stroke = color;
    svgCircle.style.strokeDashoffset = circ;
    requestAnimationFrame(() => {
        svgCircle.style.transition = 'stroke-dashoffset 1.2s ease';
        svgCircle.style.strokeDashoffset = circ * (1 - pct / 100);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    // Flash auto-dismiss
    document.querySelectorAll('.alert').forEach(el => {
        setTimeout(() => {
            el.style.transition = 'opacity 0.4s';
            el.style.opacity = '0';
            setTimeout(() => el.remove(), 400);
        }, 4000);
    });

    // Upload zone label update
    const fileInput = document.getElementById('dataset-upload');
    if (fileInput) {
        fileInput.addEventListener('change', () => {
            const label = document.getElementById('upload-label');
            if (label && fileInput.files.length) {
                label.textContent = fileInput.files[0].name;
            }
        });
    }
});
