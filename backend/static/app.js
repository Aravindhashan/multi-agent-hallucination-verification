document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('verifyForm');
    const verifyBtn = document.getElementById('verifyBtn');
    
    const idleState = document.getElementById('idleState');
    const loadingState = document.getElementById('loadingState');
    const resultsState = document.getElementById('resultsState');
    const statusBadge = document.getElementById('statusBadge');
    
    // UI Elements for results
    const hIndexValue = document.getElementById('hIndexValue');
    const hIndexLabel = document.getElementById('hIndexLabel');
    const overallVerdict = document.getElementById('overallVerdict');
    const confidenceScore = document.getElementById('confidenceScore');
    const claimsCount = document.getElementById('claimsCount');
    const evidenceCount = document.getElementById('evidenceCount');
    const processingTime = document.getElementById('processingTime');
    const claimsList = document.getElementById('claimsList');

    const sourcesContainer = document.getElementById('sourcesContainer');
    const sourcesList = document.getElementById('sourcesList');
    const sourceCount = document.getElementById('sourceCount');

    const stepText = document.getElementById('stepText');
    const liveTimer = document.getElementById('liveTimer');
    const uIndexBadge = document.getElementById('uIndexBadge');

    function escapeHtml(text) {
        if (!text) return '';
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const responseText = document.getElementById('response').value.trim();
        const mode = document.getElementById('mode').value;
        const debateRounds = parseInt(document.getElementById('debateRounds')?.value || "1", 10);

        if (!responseText) return;

        // Reset UI
        idleState.classList.add('hidden');
        resultsState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        
        verifyBtn.disabled = true;
        verifyBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Verifying...';
        statusBadge.textContent = 'Processing';
        statusBadge.className = 'px-2.5 py-1 rounded text-xs font-medium bg-blue-100 text-blue-800 uppercase tracking-wide';

        // Live stopwatch
        const startTime = performance.now();
        if (stepText) stepText.textContent = "1/3 Decomposing passage into atomic claims...";
        
        const timerInterval = setInterval(() => {
            const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
            if (liveTimer) liveTimer.textContent = `Elapsed: ${elapsed}s`;
            
            if (elapsed > 1.5 && elapsed < 8.0 && stepText) {
                stepText.textContent = `2/3 Searching internet & running debate (${mode} mode)...`;
            } else if (elapsed >= 8.0 && stepText) {
                stepText.textContent = "3/3 Cross-encoder DeBERTa-v3 adjudication...";
            }
        }, 100);

        try {
            const payload = {
                text: responseText,
                evidence_mode: mode,
                max_debate_rounds: debateRounds
            };

            const res = await fetch('/api/v1/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `Server returned ${res.status}`);
            }

            const data = await res.json();
            const endTime = performance.now();
            const durationSec = (endTime - startTime) / 1000;

            renderResults(data, durationSec.toFixed(2));
            
        } catch (error) {
            console.error('Verification error:', error);
            alert(`Failed to verify response. Error: ${error.message}`);
            loadingState.classList.add('hidden');
            idleState.classList.remove('hidden');
            statusBadge.textContent = 'Error';
            statusBadge.className = 'px-2.5 py-1 rounded text-xs font-medium bg-red-100 text-red-800 uppercase tracking-wide';
        } finally {
            clearInterval(timerInterval);
            verifyBtn.disabled = false;
            verifyBtn.innerHTML = '<i class="fas fa-magnifying-glass"></i> Verify Response';
        }
    });

    function renderResults(data, timeStr) {
        loadingState.classList.add('hidden');
        resultsState.classList.remove('hidden');
        
        statusBadge.textContent = 'Completed';
        statusBadge.className = 'px-2.5 py-1 rounded text-xs font-medium bg-green-100 text-green-800 uppercase tracking-wide';

        const hIndex = (data.hallucination_index !== undefined ? data.hallucination_index : 0).toFixed(2);
        hIndexValue.textContent = hIndex;
        
        if (hIndex <= 0.2) {
            hIndexValue.className = 'text-4xl font-bold text-green-600';
            hIndexLabel.textContent = 'Highly Factual';
            hIndexLabel.className = 'text-xs font-medium mt-1 px-2 py-0.5 rounded-full bg-green-100 text-green-800';
        } else if (hIndex <= 0.6) {
            hIndexValue.className = 'text-4xl font-bold text-yellow-500';
            hIndexLabel.textContent = 'Partial Hallucination';
            hIndexLabel.className = 'text-xs font-medium mt-1 px-2 py-0.5 rounded-full bg-yellow-100 text-yellow-800';
        } else {
            hIndexValue.className = 'text-4xl font-bold text-red-600';
            hIndexLabel.textContent = 'Severe Hallucination';
            hIndexLabel.className = 'text-xs font-medium mt-1 px-2 py-0.5 rounded-full bg-red-100 text-red-800';
        }

        const uIndex = data.unverified_index !== undefined ? data.unverified_index : 0;
        if (uIndexBadge) {
            if (uIndex > 0) {
                uIndexBadge.textContent = `Coverage Gap: ${(uIndex * 100).toFixed(0)}%`;
                uIndexBadge.classList.remove('hidden');
            } else {
                uIndexBadge.classList.add('hidden');
            }
        }

        overallVerdict.textContent = data.overall_verdict;
        if (data.overall_verdict === 'SUPPORTED') {
            overallVerdict.className = 'text-2xl font-bold text-green-600 mt-1';
        } else if (data.overall_verdict === 'REFUTED') {
            overallVerdict.className = 'text-2xl font-bold text-red-600 mt-1';
        } else {
            overallVerdict.className = 'text-2xl font-bold text-yellow-600 mt-1';
        }

        const confidence = data.confidence_score !== undefined ? data.confidence_score : 0;
        confidenceScore.textContent = `Aggregated Confidence: ${(confidence * 100).toFixed(0)}%`;
        claimsCount.textContent = data.claim_count !== undefined ? data.claim_count : (data.claims || []).length;
        
        // 1. Render Aggregated Internet Sources Panel
        const internetSources = data.internet_sources || [];
        if (sourcesContainer && sourcesList) {
            if (internetSources.length > 0) {
                sourcesContainer.classList.remove('hidden');
                if (sourceCount) sourceCount.textContent = internetSources.length;
                sourcesList.innerHTML = internetSources.map(src => {
                    const title = escapeHtml(src.title || 'Internet Source');
                    const domain = escapeHtml(src.domain || 'web');
                    const snippet = escapeHtml(src.snippet || '');
                    const url = src.url;
                    
                    const titleHtml = url 
                        ? `<a href="${encodeURI(url)}" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:text-blue-800 font-semibold hover:underline flex items-center gap-1.5 truncate"><i class="fas fa-arrow-up-right-from-square text-[10px]"></i> ${title}</a>`
                        : `<span class="font-semibold text-gray-800 truncate">${title}</span>`;
                    
                    return `
                        <div class="bg-white border border-blue-200/80 rounded-lg p-3 shadow-xs flex flex-col justify-between">
                            <div class="flex items-center justify-between gap-2 mb-1.5">
                                <div class="truncate text-xs">${titleHtml}</div>
                                <span class="text-[10px] font-mono bg-blue-100/80 text-blue-800 px-2 py-0.5 rounded shrink-0">${domain}</span>
                            </div>
                            ${snippet ? `<p class="text-[11px] text-gray-600 line-clamp-2 italic">"${snippet}"</p>` : ''}
                        </div>
                    `;
                }).join('');
            } else {
                sourcesContainer.classList.add('hidden');
            }
        }

        // 2. Render Atomic Claims Breakdown with Supportive & Opposing Evidence
        let totalEvidences = 0;
        claimsList.innerHTML = '';

        if (data.claims && data.claims.length > 0) {
            data.claims.forEach((claim, idx) => {
                const supportive = claim.supportive_claims || [];
                const opposing = claim.opposing_claims || [];
                const claimSources = claim.internet_sources || [];
                const legacySnippets = claim.winning_evidence_snippets || [];
                
                totalEvidences += (supportive.length + opposing.length) || legacySnippets.length;

                // Build supportive cards
                let supportiveHTML = '';
                if (supportive.length > 0) {
                    supportiveHTML = supportive.map(ev => {
                        const sourceLink = ev.source_url 
                            ? `<a href="${encodeURI(ev.source_url)}" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:underline truncate">${escapeHtml(ev.source_title || ev.domain || 'Internet Source')}</a>`
                            : `<span class="truncate">${escapeHtml(ev.source_title || ev.domain || 'Source')}</span>`;
                        const confText = ev.confidence !== null && ev.confidence !== undefined 
                            ? `${Math.round(ev.confidence * 100)}% Entails` 
                            : 'Supportive';
                        const nliMetaBadge = (ev.p_entailment !== null && ev.p_entailment !== undefined)
                            ? `<div class="mt-2 text-[10px] text-gray-500 font-mono bg-gray-50 border border-gray-200 rounded px-1.5 py-0.5 flex flex-wrap gap-2 items-center">
                                 <span><i class="fas fa-microchip text-indigo-500 mr-1"></i>DeBERTa: <b>${ev.token_length || '~'} tokens</b></span>
                                 <span class="text-green-700 font-semibold">E: ${Math.round(ev.p_entailment * 100)}%</span>
                                 <span class="text-yellow-700 font-semibold">N: ${Math.round((ev.p_neutral || 0) * 100)}%</span>
                                 <span class="text-red-700 font-semibold">C: ${Math.round((ev.p_contradiction || 0) * 100)}%</span>
                               </div>`
                            : '';

                        return `
                            <div class="bg-white border border-green-200 rounded-lg p-3 text-xs shadow-xs">
                                <p class="text-gray-800 leading-relaxed font-normal">"${escapeHtml(ev.text)}"</p>
                                ${nliMetaBadge}
                                <div class="mt-2.5 flex items-center justify-between text-[11px] pt-1.5 border-t border-gray-100">
                                    <div class="flex items-center gap-1.5 text-gray-600 truncate mr-2">
                                        <i class="fas fa-book-bookmark text-green-600"></i>
                                        ${sourceLink}
                                    </div>
                                    <span class="bg-green-100 text-green-800 font-bold px-1.5 py-0.5 rounded shrink-0">${confText}</span>
                                </div>
                            </div>
                        `;
                    }).join('');
                } else if (legacySnippets.length > 0 && opposing.length === 0) {
                    // Fallback for legacy items
                    supportiveHTML = legacySnippets.map(snip => `
                        <div class="bg-white border border-green-200 rounded-lg p-3 text-xs shadow-xs">
                            <p class="text-gray-800 leading-relaxed">"${escapeHtml(snip)}"</p>
                            <div class="mt-2 text-[10px] text-gray-500 flex items-center gap-1">
                                <i class="fas fa-check text-green-600"></i> Representative Passage
                            </div>
                        </div>
                    `).join('');
                } else {
                    supportiveHTML = `
                        <div class="text-xs text-gray-500 italic p-3 text-center bg-white/70 rounded-lg border border-dashed border-gray-200 flex items-center justify-center gap-1.5">
                            <i class="fas fa-info-circle text-gray-400"></i> No supportive claims or passages found.
                        </div>
                    `;
                }

                // Build opposing cards
                let opposingHTML = '';
                if (opposing.length > 0) {
                    opposingHTML = opposing.map(ev => {
                        const sourceLink = ev.source_url 
                            ? `<a href="${encodeURI(ev.source_url)}" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:underline truncate">${escapeHtml(ev.source_title || ev.domain || 'Internet Source')}</a>`
                            : `<span class="truncate">${escapeHtml(ev.source_title || ev.domain || 'Source')}</span>`;
                        const confText = ev.confidence !== null && ev.confidence !== undefined 
                            ? `${Math.round(ev.confidence * 100)}% Refutes` 
                            : 'Opposing';
                        const nliMetaBadge = (ev.p_contradiction !== null && ev.p_contradiction !== undefined)
                            ? `<div class="mt-2 text-[10px] text-gray-500 font-mono bg-gray-50 border border-gray-200 rounded px-1.5 py-0.5 flex flex-wrap gap-2 items-center">
                                 <span><i class="fas fa-microchip text-indigo-500 mr-1"></i>DeBERTa: <b>${ev.token_length || '~'} tokens</b></span>
                                 <span class="text-green-700 font-semibold">E: ${Math.round((ev.p_entailment || 0) * 100)}%</span>
                                 <span class="text-yellow-700 font-semibold">N: ${Math.round((ev.p_neutral || 0) * 100)}%</span>
                                 <span class="text-red-700 font-semibold">C: ${Math.round(ev.p_contradiction * 100)}%</span>
                               </div>`
                            : '';

                        return `
                            <div class="bg-white border border-red-200 rounded-lg p-3 text-xs shadow-xs">
                                <p class="text-gray-800 leading-relaxed font-normal">"${escapeHtml(ev.text)}"</p>
                                ${nliMetaBadge}
                                <div class="mt-2.5 flex items-center justify-between text-[11px] pt-1.5 border-t border-gray-100">
                                    <div class="flex items-center gap-1.5 text-gray-600 truncate mr-2">
                                        <i class="fas fa-triangle-exclamation text-red-500"></i>
                                        ${sourceLink}
                                    </div>
                                    <span class="bg-red-100 text-red-800 font-bold px-1.5 py-0.5 rounded shrink-0">${confText}</span>
                                </div>
                            </div>
                        `;
                    }).join('');
                } else {
                    opposingHTML = `
                        <div class="text-xs text-gray-500 italic p-3 text-center bg-white/70 rounded-lg border border-dashed border-gray-200 flex items-center justify-center gap-1.5">
                            <i class="fas fa-shield-halved text-gray-400"></i> No opposing claims found on the web.
                        </div>
                    `;
                }

                // Build claim sources row
                let claimSourcesHTML = '';
                if (claimSources.length > 0) {
                    const sourceBadges = claimSources.map(s => {
                        const sTitle = escapeHtml(s.title || s.domain || 'Web Source');
                        const sDomain = escapeHtml(s.domain || 'web');
                        if (s.url) {
                            return `<a href="${encodeURI(s.url)}" target="_blank" rel="noopener noreferrer" class="inline-flex items-center gap-1.5 text-[11px] text-blue-700 hover:text-blue-900 bg-blue-50/80 hover:bg-blue-100 border border-blue-200 px-2 py-0.5 rounded-full transition-colors font-medium">
                                <i class="fas fa-link text-[9px]"></i> <span class="max-w-[180px] truncate">${sTitle}</span> <span class="text-[9px] text-blue-500 font-mono">(${sDomain})</span>
                            </a>`;
                        }
                        return `<span class="inline-flex items-center gap-1.5 text-[11px] text-gray-700 bg-gray-100 border border-gray-200 px-2 py-0.5 rounded-full font-medium">
                            <i class="fas fa-file-lines text-[9px]"></i> <span class="max-w-[180px] truncate">${sTitle}</span>
                        </span>`;
                    }).join('');

                    claimSourcesHTML = `
                        <div class="mt-4 pt-3 border-t border-gray-100">
                            <div class="text-[11px] font-bold text-gray-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                                <i class="fas fa-globe text-primary"></i> Internet Sources for Claim ${idx + 1} (${claimSources.length})
                            </div>
                            <div class="flex flex-wrap gap-1.5">
                                ${sourceBadges}
                            </div>
                        </div>
                    `;
                }

                let badgeClass = 'bg-gray-100 text-gray-800 border-gray-200';
                if (claim.verdict === 'SUPPORTED') badgeClass = 'bg-green-100 text-green-800 border-green-200';
                if (claim.verdict === 'REFUTED') badgeClass = 'bg-red-100 text-red-800 border-red-200';
                if (claim.verdict === 'NOT_ENOUGH_INFO') badgeClass = 'bg-yellow-100 text-yellow-800 border-yellow-200';

                const claimConfidence = claim.confidence_score !== undefined ? claim.confidence_score : 0;
                const entailment = claim.support_probability !== undefined ? claim.support_probability : 0;
                const contradiction = claim.contradiction_probability !== undefined ? claim.contradiction_probability : 0;

                const claimHTML = `
                    <div class="bg-white border border-gray-200 rounded-xl p-5 claim-card shadow-sm">
                        <!-- Claim Title Header -->
                        <div class="flex justify-between items-start gap-4 pb-3 border-b border-gray-100">
                            <div class="flex-grow">
                                <div class="flex items-center gap-2 mb-1.5">
                                    <span class="text-xs font-bold text-gray-400">CLAIM ${idx + 1}</span>
                                    <span class="px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase border ${badgeClass}">
                                        ${claim.verdict}
                                    </span>
                                    <span class="text-xs text-gray-500 font-medium">Confidence: ${(claimConfidence * 100).toFixed(0)}%</span>
                                </div>
                                <p class="text-gray-900 font-semibold text-base">${escapeHtml(claim.claim_text)}</p>
                                <div class="mt-1.5 text-xs text-gray-500 flex items-center gap-4">
                                    <span class="flex items-center gap-1"><i class="fas fa-microchip text-primary"></i> NLI Entailment: ${(entailment * 100).toFixed(1)}% | Contradiction: ${(contradiction * 100).toFixed(1)}%</span>
                                </div>
                            </div>
                        </div>

                        <!-- Dual Column: Supportive vs Opposing Claims -->
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                            <!-- Supportive Claims Column -->
                            <div class="bg-green-50/40 border border-green-200/80 rounded-xl p-3.5 flex flex-col">
                                <div class="flex items-center justify-between pb-2 mb-2.5 border-b border-green-200">
                                    <span class="text-xs font-bold text-green-900 uppercase flex items-center gap-1.5">
                                        <i class="fas fa-circle-check text-green-600"></i> Supportive Claims & Evidence
                                    </span>
                                    <span class="text-[10px] font-bold bg-green-200 text-green-900 px-2 py-0.5 rounded-full">
                                        ${supportive.length}
                                    </span>
                                </div>
                                <div class="flex flex-col gap-2.5 flex-grow">
                                    ${supportiveHTML}
                                </div>
                            </div>

                            <!-- Opposing Claims Column -->
                            <div class="bg-red-50/40 border border-red-200/80 rounded-xl p-3.5 flex flex-col">
                                <div class="flex items-center justify-between pb-2 mb-2.5 border-b border-red-200">
                                    <span class="text-xs font-bold text-red-900 uppercase flex items-center gap-1.5">
                                        <i class="fas fa-circle-xmark text-red-600"></i> Opposing Claims & Evidence
                                    </span>
                                    <span class="text-[10px] font-bold bg-red-200 text-red-900 px-2 py-0.5 rounded-full">
                                        ${opposing.length}
                                    </span>
                                </div>
                                <div class="flex flex-col gap-2.5 flex-grow">
                                    ${opposingHTML}
                                </div>
                            </div>
                        </div>

                        <!-- Internet Sources for this claim -->
                        ${claimSourcesHTML}
                    </div>
                `;
                
                claimsList.insertAdjacentHTML('beforeend', claimHTML);
            });
        }

        evidenceCount.textContent = data.total_evidence_evaluated !== undefined ? data.total_evidence_evaluated : totalEvidences;
        processingTime.textContent = (data.execution_time_ms ? (data.execution_time_ms / 1000).toFixed(2) : timeStr) + 's';
    }

    // Quick demo preset button handlers
    const btnSampleFact = document.getElementById('btnSampleFact');
    const btnSampleHallucination = document.getElementById('btnSampleHallucination');
    const btnSamplePCSK9 = document.getElementById('btnSamplePCSK9');
    const responseInput = document.getElementById('response');
    const queryInput = document.getElementById('query');
    const modeSelect = document.getElementById('mode');

    if (btnSampleFact && responseInput) {
        btnSampleFact.addEventListener('click', () => {
            if (queryInput) queryInput.value = "What is the cardiovascular efficacy of Aspirin?";
            responseInput.value = "Aspirin reduces cardiovascular mortality and lowers the risk of myocardial infarction in clinical patients.";
            if (modeSelect) modeSelect.value = "live_web";
        });
    }

    if (btnSampleHallucination && responseInput) {
        btnSampleHallucination.addEventListener('click', () => {
            if (queryInput) queryInput.value = "Can high doses of Vitamin C cure viral infections?";
            responseInput.value = "High-dose Vitamin C cures acute viral hepatitis and completely prevents influenza virus transmission in humans.";
            if (modeSelect) modeSelect.value = "live_web";
        });
    }

    if (btnSamplePCSK9 && responseInput) {
        btnSamplePCSK9.addEventListener('click', () => {
            if (queryInput) queryInput.value = "What is the genetic and clinical role of PCSK9 in hypercholesterolemia?";
            responseInput.value = "Loss-of-function mutations in the PCSK9 gene cause severe hypercholesterolemia and coronary artery disease.";
            if (modeSelect) modeSelect.value = "live_web";
        });
    }
});
