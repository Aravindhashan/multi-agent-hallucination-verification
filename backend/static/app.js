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
    const charCounter = document.getElementById('charCounter');
    const responseInput = document.getElementById('response');
    const queryInput = document.getElementById('query');
    const modeSelect = document.getElementById('mode');

    // Preset Buttons
    const btnSampleFact = document.getElementById('btnSampleFact');
    const btnSampleHallucination = document.getElementById('btnSampleHallucination');
    const btnSamplePCSK9 = document.getElementById('btnSamplePCSK9');
    const btnSampleList = document.getElementById('btnSampleList');

    function escapeHtml(text) {
        if (!text) return '';
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // Dynamic character counter
    if (responseInput && charCounter) {
        responseInput.addEventListener('input', () => {
            charCounter.textContent = `${responseInput.value.length} chars`;
        });
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const responseText = responseInput.value.trim();
        const mode = modeSelect.value;
        const debateRounds = parseInt(document.getElementById('debateRounds')?.value || "1", 10);

        if (!responseText) return;

        // Transition UI to loading state
        idleState.classList.add('hidden');
        resultsState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        
        verifyBtn.disabled = true;
        verifyBtn.innerHTML = '<i class="fas fa-spinner fa-spin text-sm"></i><span class="tracking-wide text-sm">Adjudicating...</span>';
        statusBadge.textContent = 'Adjudicating';
        statusBadge.className = 'px-3 py-1 rounded-full text-[11px] font-mono font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 uppercase tracking-wider animate-pulse';

        // Live stopwatch and progress messaging
        const startTime = performance.now();
        if (stepText) stepText.textContent = "1/3 Decomposing passage into atomic claims...";
        
        const timerInterval = setInterval(() => {
            const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
            if (liveTimer) liveTimer.textContent = `Elapsed: ${elapsed}s`;
            
            if (elapsed > 1.5 && elapsed < 8.0 && stepText) {
                stepText.textContent = `2/3 Querying federated evidence & multi-agent debate (${mode} mode)...`;
            } else if (elapsed >= 8.0 && stepText) {
                stepText.textContent = "3/3 Cross-encoder DeBERTa-v3 adjudication & calibration...";
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
                throw new Error(errData.detail || `Server returned status ${res.status}`);
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
            statusBadge.className = 'px-3 py-1 rounded-full text-[11px] font-mono font-semibold bg-rose-500/20 text-rose-300 border border-rose-500/30 uppercase tracking-wider';
        } finally {
            clearInterval(timerInterval);
            verifyBtn.disabled = false;
            verifyBtn.innerHTML = '<i class="fas fa-magnifying-glass text-sm"></i><span class="tracking-wide text-sm">Verify Response</span>';
        }
    });

    function renderResults(data, timeStr) {
        loadingState.classList.add('hidden');
        resultsState.classList.remove('hidden');
        
        statusBadge.textContent = 'Verified';
        statusBadge.className = 'px-3 py-1 rounded-full text-[11px] font-mono font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 uppercase tracking-wider';

        const hIndex = (data.hallucination_index !== undefined ? data.hallucination_index : 0).toFixed(2);
        hIndexValue.textContent = hIndex;
        
        const uIndex = data.unverified_index !== undefined ? data.unverified_index : 0;
        if (hIndex <= 0.2) {
            hIndexValue.className = 'text-4xl font-extrabold tracking-tight font-mono text-emerald-400';
            if (uIndex > 0) {
                hIndexLabel.textContent = 'Factual (Coverage Gap)';
                hIndexLabel.className = 'text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-emerald-950/80 text-emerald-300 border border-emerald-500/30';
            } else {
                hIndexLabel.textContent = 'Highly Factual';
                hIndexLabel.className = 'text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-emerald-950/80 text-emerald-300 border border-emerald-500/30';
            }
        } else if (hIndex <= 0.6) {
            hIndexValue.className = 'text-4xl font-extrabold tracking-tight font-mono text-amber-400';
            hIndexLabel.textContent = 'Partial Hallucination';
            hIndexLabel.className = 'text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-amber-950/80 text-amber-300 border border-amber-500/30';
        } else {
            hIndexValue.className = 'text-4xl font-extrabold tracking-tight font-mono text-rose-400';
            hIndexLabel.textContent = 'Severe Hallucination';
            hIndexLabel.className = 'text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-rose-950/80 text-rose-300 border border-rose-500/30';
        }

        if (uIndexBadge) {
            if (uIndex > 0) {
                uIndexBadge.textContent = `Coverage Gap: ${(uIndex * 100).toFixed(0)}%`;
                uIndexBadge.className = 'text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-amber-950/80 text-amber-300 border border-amber-500/30';
                uIndexBadge.classList.remove('hidden');
            } else {
                uIndexBadge.classList.add('hidden');
            }
        }

        const refCount = data.refuted_count || 0;
        const supCount = data.supported_count || 0;
        const neiCount = data.unverified_count || 0;
        const vCat = data.verdict_category || (
            (data.overall_verdict === 'PARTIALLY_SUPPORTED' && refCount === 0) 
                ? 'PARTIALLY_VERIFIED_COVERAGE_GAP' 
                : (data.overall_verdict === 'PARTIALLY_SUPPORTED' && refCount > 0)
                    ? 'PARTIALLY_SUPPORTED_CONTRADICTED'
                    : data.overall_verdict
        );

        let verdictContext = '';
        if (vCat === 'SUPPORTED' || data.overall_verdict === 'SUPPORTED') {
            overallVerdict.textContent = 'SUPPORTED';
            overallVerdict.className = 'text-2xl font-black tracking-wide text-emerald-400 mt-0.5';
            verdictContext = ` · Verified (${supCount} Supported, 0 Refuted)`;
        } else if (vCat === 'PARTIALLY_VERIFIED_COVERAGE_GAP') {
            overallVerdict.textContent = 'PARTIALLY VERIFIED';
            overallVerdict.className = 'text-2xl font-black tracking-wide text-teal-400 mt-0.5';
            verdictContext = ` · 0 Contradictions (${supCount} Supported, ${neiCount} Coverage Gap)`;
        } else if (vCat === 'PARTIALLY_SUPPORTED_CONTRADICTED') {
            overallVerdict.textContent = 'PARTIALLY SUPPORTED';
            overallVerdict.className = 'text-2xl font-black tracking-wide text-amber-400 mt-0.5';
            verdictContext = ` · Contradictions Detected (${supCount} Supported, ${refCount} Refuted)`;
        } else if (vCat === 'REFUTED' || data.overall_verdict === 'REFUTED') {
            overallVerdict.textContent = 'REFUTED';
            overallVerdict.className = 'text-2xl font-black tracking-wide text-rose-400 mt-0.5';
            verdictContext = ` · Contradicted (${refCount} Refuted, 0 Supported)`;
        } else {
            overallVerdict.textContent = 'NOT ENOUGH INFO';
            overallVerdict.className = 'text-2xl font-black tracking-wide text-slate-400 mt-0.5';
            verdictContext = ` · Inconclusive (${neiCount} Unverified)`;
        }

        const confidence = data.confidence_score !== undefined ? data.confidence_score : 0;
        confidenceScore.textContent = `Aggregated Confidence: ${(confidence * 100).toFixed(0)}%${verdictContext}`;
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
                        ? `<a href="${encodeURI(url)}" target="_blank" rel="noopener noreferrer" class="text-indigo-400 hover:text-indigo-300 font-semibold hover:underline flex items-center gap-1.5 truncate"><i class="fas fa-arrow-up-right-from-square text-[10px]"></i> ${title}</a>`
                        : `<span class="font-semibold text-slate-200 truncate">${title}</span>`;
                    
                    return `
                        <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-3 shadow-md flex flex-col justify-between">
                            <div class="flex items-center justify-between gap-2 mb-1.5">
                                <div class="truncate text-xs">${titleHtml}</div>
                                <span class="text-[10px] font-mono bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 px-2 py-0.5 rounded shrink-0">${domain}</span>
                            </div>
                            ${snippet ? `<p class="text-[11px] text-slate-400 line-clamp-2 italic">"${snippet}"</p>` : ''}
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
                            ? `<a href="${encodeURI(ev.source_url)}" target="_blank" rel="noopener noreferrer" class="text-emerald-400 hover:text-emerald-300 hover:underline truncate">${escapeHtml(ev.source_title || ev.domain || 'Internet Source')}</a>`
                            : `<span class="truncate text-slate-300">${escapeHtml(ev.source_title || ev.domain || 'Source')}</span>`;
                        const confText = ev.confidence !== null && ev.confidence !== undefined 
                            ? `${Math.round(ev.confidence * 100)}% Entails` 
                            : 'Supportive';
                        const nliMetaBadge = (ev.p_entailment !== null && ev.p_entailment !== undefined)
                            ? `<div class="mt-2 text-[10px] text-slate-400 font-mono bg-slate-900/90 border border-slate-800 rounded-lg px-2 py-1 flex flex-wrap gap-2 items-center">
                                 <span><i class="fas fa-microchip text-indigo-400 mr-1"></i>DeBERTa: <b>${ev.token_length || '~'} tokens</b></span>
                                 <span class="text-emerald-400 font-semibold">E: ${Math.round(ev.p_entailment * 100)}%</span>
                                 <span class="text-amber-400 font-semibold">N: ${Math.round((ev.p_neutral || 0) * 100)}%</span>
                                 <span class="text-rose-400 font-semibold">C: ${Math.round((ev.p_contradiction || 0) * 100)}%</span>
                               </div>`
                            : '';

                        return `
                            <div class="bg-slate-900/80 border border-emerald-500/30 rounded-xl p-3 text-xs shadow-md">
                                <p class="text-slate-200 leading-relaxed font-normal">"${escapeHtml(ev.text)}"</p>
                                ${nliMetaBadge}
                                <div class="mt-2.5 flex items-center justify-between text-[11px] pt-2 border-t border-slate-800">
                                    <div class="flex items-center gap-1.5 text-slate-400 truncate mr-2">
                                        <i class="fas fa-book-bookmark text-emerald-400"></i>
                                        ${sourceLink}
                                    </div>
                                    <span class="bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-mono font-bold px-2 py-0.5 rounded-full shrink-0">${confText}</span>
                                </div>
                            </div>
                        `;
                    }).join('');
                } else if (legacySnippets.length > 0 && opposing.length === 0) {
                    supportiveHTML = legacySnippets.map(snip => `
                        <div class="bg-slate-900/80 border border-emerald-500/30 rounded-xl p-3 text-xs shadow-md">
                            <p class="text-slate-200 leading-relaxed font-normal">"${escapeHtml(snip)}"</p>
                            <div class="mt-2 text-[10px] font-mono text-slate-400 flex items-center gap-1">
                                <i class="fas fa-check text-emerald-400"></i> Representative Corroborating Passage
                            </div>
                        </div>
                    `).join('');
                } else {
                    supportiveHTML = `
                        <div class="text-xs text-slate-500 italic p-4 text-center bg-slate-900/40 rounded-xl border border-dashed border-slate-800 flex items-center justify-center gap-2">
                            <i class="fas fa-info-circle text-slate-500"></i> No confirming passages retrieved.
                        </div>
                    `;
                }

                // Build opposing cards
                let opposingHTML = '';
                if (opposing.length > 0) {
                    opposingHTML = opposing.map(ev => {
                        const sourceLink = ev.source_url 
                            ? `<a href="${encodeURI(ev.source_url)}" target="_blank" rel="noopener noreferrer" class="text-rose-400 hover:text-rose-300 hover:underline truncate">${escapeHtml(ev.source_title || ev.domain || 'Internet Source')}</a>`
                            : `<span class="truncate text-slate-300">${escapeHtml(ev.source_title || ev.domain || 'Source')}</span>`;
                        const confText = ev.confidence !== null && ev.confidence !== undefined 
                            ? `${Math.round(ev.confidence * 100)}% Refutes` 
                            : 'Opposing';
                        const nliMetaBadge = (ev.p_contradiction !== null && ev.p_contradiction !== undefined)
                            ? `<div class="mt-2 text-[10px] text-slate-400 font-mono bg-slate-900/90 border border-slate-800 rounded-lg px-2 py-1 flex flex-wrap gap-2 items-center">
                                 <span><i class="fas fa-microchip text-indigo-400 mr-1"></i>DeBERTa: <b>${ev.token_length || '~'} tokens</b></span>
                                 <span class="text-emerald-400 font-semibold">E: ${Math.round((ev.p_entailment || 0) * 100)}%</span>
                                 <span class="text-amber-400 font-semibold">N: ${Math.round((ev.p_neutral || 0) * 100)}%</span>
                                 <span class="text-rose-400 font-semibold">C: ${Math.round(ev.p_contradiction * 100)}%</span>
                               </div>`
                            : '';

                        return `
                            <div class="bg-slate-900/80 border border-rose-500/30 rounded-xl p-3 text-xs shadow-md">
                                <p class="text-slate-200 leading-relaxed font-normal">"${escapeHtml(ev.text)}"</p>
                                ${nliMetaBadge}
                                <div class="mt-2.5 flex items-center justify-between text-[11px] pt-2 border-t border-slate-800">
                                    <div class="flex items-center gap-1.5 text-slate-400 truncate mr-2">
                                        <i class="fas fa-triangle-exclamation text-rose-400"></i>
                                        ${sourceLink}
                                    </div>
                                    <span class="bg-rose-500/20 text-rose-300 border border-rose-500/30 font-mono font-bold px-2 py-0.5 rounded-full shrink-0">${confText}</span>
                                </div>
                            </div>
                        `;
                    }).join('');
                } else {
                    const isConsensus = claim.consensus_confirmed || claim.verdict === 'SUPPORTED';
                    if (isConsensus) {
                        opposingHTML = `
                            <div class="bg-emerald-950/40 border border-emerald-500/40 rounded-xl p-3.5 text-xs text-emerald-200 shadow-md flex items-start gap-2.5">
                                <i class="fas fa-circle-check text-emerald-400 text-sm shrink-0 mt-0.5"></i>
                                <div>
                                    <span class="font-bold text-emerald-300">✓ Scientific Consensus Confirmed:</span>
                                    <span class="text-emerald-200/90"> Exhaustive adversarial prosecutor search yielded 0 contradictory claims.</span>
                                </div>
                            </div>
                        `;
                    } else {
                        opposingHTML = `
                            <div class="text-xs text-slate-500 italic p-4 text-center bg-slate-900/40 rounded-xl border border-dashed border-slate-800 flex items-center justify-center gap-2">
                                <i class="fas fa-shield-halved text-slate-500"></i> No opposing claims found on the web.
                            </div>
                        `;
                    }
                }

                // Build claim sources row
                let claimSourcesHTML = '';
                if (claimSources.length > 0) {
                    const sourceBadges = claimSources.map(s => {
                        const sTitle = escapeHtml(s.title || s.domain || 'Web Source');
                        const sDomain = escapeHtml(s.domain || 'web');
                        if (s.url) {
                            return `<a href="${encodeURI(s.url)}" target="_blank" rel="noopener noreferrer" class="inline-flex items-center gap-1.5 text-[11px] text-indigo-300 hover:text-indigo-200 bg-indigo-500/10 hover:bg-indigo-500/20 border border-indigo-500/20 px-2.5 py-1 rounded-full transition-all font-sans">
                                <i class="fas fa-link text-[9px] text-indigo-400"></i> <span class="max-w-[180px] truncate">${sTitle}</span> <span class="text-[9px] text-indigo-400 font-mono">(${sDomain})</span>
                            </a>`;
                        }
                        return `<span class="inline-flex items-center gap-1.5 text-[11px] text-slate-300 bg-slate-800/80 border border-slate-700/60 px-2.5 py-1 rounded-full font-sans">
                            <i class="fas fa-file-lines text-[9px] text-slate-400"></i> <span class="max-w-[180px] truncate">${sTitle}</span>
                        </span>`;
                    }).join('');

                    claimSourcesHTML = `
                        <div class="mt-4 pt-3 border-t border-slate-800">
                            <div class="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                                <i class="fas fa-globe text-indigo-400"></i> Provenance Links for Claim ${idx + 1} (${claimSources.length})
                            </div>
                            <div class="flex flex-wrap gap-1.5">
                                ${sourceBadges}
                            </div>
                        </div>
                    `;
                }

                let badgeClass = 'bg-slate-800 text-slate-300 border-slate-700';
                if (claim.verdict === 'SUPPORTED') badgeClass = 'bg-emerald-950/80 text-emerald-300 border-emerald-500/40';
                if (claim.verdict === 'REFUTED') badgeClass = 'bg-rose-950/80 text-rose-300 border-rose-500/40';
                if (claim.verdict === 'NOT_ENOUGH_INFO') badgeClass = 'bg-amber-950/80 text-amber-300 border-amber-500/40';

                const claimConfidence = claim.confidence_score !== undefined ? claim.confidence_score : 0;
                const entailment = claim.support_probability !== undefined ? claim.support_probability : 0;
                const contradiction = claim.contradiction_probability !== undefined ? claim.contradiction_probability : 0;

                const claimHTML = `
                    <div class="glass-card rounded-2xl p-5 border border-slate-800 shadow-xl bg-slate-900/60">
                        <!-- Claim Title Header -->
                        <div class="flex justify-between items-start gap-4 pb-3 border-b border-slate-800">
                            <div class="flex-grow">
                                <div class="flex items-center gap-2 mb-1.5">
                                    <span class="text-xs font-mono font-bold text-slate-400">CLAIM ${idx + 1}</span>
                                    <span class="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold uppercase border ${badgeClass}">
                                        ${claim.verdict}
                                    </span>
                                    <span class="text-xs font-mono text-slate-400">Confidence: ${(claimConfidence * 100).toFixed(0)}%</span>
                                </div>
                                <p class="text-slate-100 font-semibold text-base leading-snug">${escapeHtml(claim.claim_text)}</p>
                                <div class="mt-2 text-xs font-mono text-slate-400 flex items-center gap-4">
                                    <span class="flex items-center gap-1.5"><i class="fas fa-microchip text-indigo-400"></i> Entailment: <b class="text-emerald-400">${(entailment * 100).toFixed(1)}%</b> | Contradiction: <b class="text-rose-400">${(contradiction * 100).toFixed(1)}%</b></span>
                                </div>
                            </div>
                        </div>

                        <!-- Dual Column: Supportive vs Opposing Claims -->
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                            <!-- Supportive Claims Column -->
                            <div class="bg-emerald-950/20 border border-emerald-500/20 rounded-xl p-3.5 flex flex-col">
                                <div class="flex items-center justify-between pb-2 mb-2.5 border-b border-emerald-500/20">
                                    <span class="text-xs font-bold text-emerald-300 uppercase tracking-wider flex items-center gap-1.5">
                                        <i class="fas fa-shield-halved text-emerald-400"></i> Defender Corroboration
                                    </span>
                                    <span class="text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-2 py-0.5 rounded-full">
                                        ${supportive.length}
                                    </span>
                                </div>
                                <div class="flex flex-col gap-2.5 flex-grow">
                                    ${supportiveHTML}
                                </div>
                            </div>

                            <!-- Opposing Claims Column -->
                            <div class="bg-rose-950/20 border border-rose-500/20 rounded-xl p-3.5 flex flex-col">
                                <div class="flex items-center justify-between pb-2 mb-2.5 border-b border-rose-500/20">
                                    <span class="text-xs font-bold text-rose-300 uppercase tracking-wider flex items-center gap-1.5">
                                        <i class="fas fa-scale-balanced text-rose-400"></i> Prosecutor Counter-Evidence
                                    </span>
                                    <span class="text-[10px] font-mono font-bold bg-rose-500/20 text-rose-300 border border-rose-500/30 px-2 py-0.5 rounded-full">
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
    if (btnSampleFact && responseInput) {
        btnSampleFact.addEventListener('click', () => {
            if (queryInput) queryInput.value = "What is the cardiovascular efficacy of Aspirin?";
            responseInput.value = "Aspirin reduces cardiovascular mortality and lowers the risk of myocardial infarction in clinical patients.";
            if (charCounter) charCounter.textContent = `${responseInput.value.length} chars`;
            if (modeSelect) modeSelect.value = "live_web";
        });
    }

    if (btnSampleHallucination && responseInput) {
        btnSampleHallucination.addEventListener('click', () => {
            if (queryInput) queryInput.value = "Can high doses of Vitamin C cure viral infections?";
            responseInput.value = "High-dose Vitamin C cures acute viral hepatitis and completely prevents influenza virus transmission in humans.";
            if (charCounter) charCounter.textContent = `${responseInput.value.length} chars`;
            if (modeSelect) modeSelect.value = "live_web";
        });
    }

    if (btnSamplePCSK9 && responseInput) {
        btnSamplePCSK9.addEventListener('click', () => {
            if (queryInput) queryInput.value = "What is the genetic and clinical role of PCSK9 in hypercholesterolemia?";
            responseInput.value = "Loss-of-function mutations in the PCSK9 gene cause severe hypercholesterolemia and coronary artery disease.";
            if (charCounter) charCounter.textContent = `${responseInput.value.length} chars`;
            if (modeSelect) modeSelect.value = "live_web";
        });
    }

    if (btnSampleList && responseInput) {
        btnSampleList.addEventListener('click', () => {
            if (queryInput) queryInput.value = "What is Nateglinide used for?";
            responseInput.value = "* Nateglinide.\n* Used for the treatment of type 2 diabetes mellitus.";
            if (charCounter) charCounter.textContent = `${responseInput.value.length} chars`;
            if (modeSelect) modeSelect.value = "live_web";
        });
    }
});
