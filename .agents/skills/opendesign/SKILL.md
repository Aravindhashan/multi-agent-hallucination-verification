---
name: opendesign
description: Design and enhance web prototypes, dashboards, interactive reports, and modern frontend interfaces using OpenDesign principles (curated design systems, visual hierarchy, polished typography, responsive layouts, and interactive components). Use whenever improving, styling, or creating frontend UIs.
triggers:
  - "opendesign"
  - "improve frontend"
  - "modern ui"
  - "dashboard design"
  - "redesign"
  - "ui polish"
---

# OpenDesign Frontend Engineering & Design Skill

Integrates open-source OpenDesign systems, UI token standards, and modern frontend playbooks directly into the workspace to create brand-grade, polished web applications and analytical dashboards.

## Core Design Principles

1. **Restrained, Professional Visual Hierarchy**:
   - Deep neutral backgrounds (`#0a0f1d`, `#0f172a`, `#1e293b`) with high-contrast text (`#f8fafc`, `#e2e8f0`).
   - Clean semantic accent colors:
     - Verified / Entailment: Emerald (`#10b981`, `#059669`)
     - Refuted / Contradiction: Rose / Crimson (`#f43f5e`, `#e11d48`)
     - Inconclusive / Neutral: Amber / Slate (`#f59e0b`, `#64748b`)
     - Primary Action / Accent: Indigo / Violet (`#6366f1`, `#8b5cf6`)
   - Avoid generic, flat monochrome. Use subtle multi-layered drop shadows, backdrop-filter blurs (`backdrop-blur-md`), and 1px border glows (`border border-slate-700/60`).

2. **Information-Dense, Card-Based Component Architecture**:
   - Every metric card should feature clear typography, badges, and contextual metadata.
   - Interactive accordions with smooth CSS transitions (`transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1)`).
   - Micro-interactions: hover highlights, status pills with pulsing indicators, and clipboard copy buttons.

3. **Responsive & Fluid Layout**:
   - Standard 1280px-1440px desktop container with fluid CSS grid (`grid-cols-1 md:grid-cols-2 lg:grid-cols-3`).
   - Clean spacing scale: 4px, 8px, 12px, 16px, 24px, 32px, 48px.
   - Distinctive typography using modern variable fonts (Inter, Plus Jakarta Sans, Outfit, JetBrains Mono for code/metrics).

4. **Component Patterns**:
   - **Hero Verification Console**: High-impact text area with sample prompts, character counter, model configuration toggles, and animated "Verify Response" CTA.
   - **Verdict Status Banner**: Dynamic color-coded gradient card displaying Global Verdict, Hallucination Index ($H$), and Epistemic Coverage Gap ($U$).
   - **Dual-Perspective Debate Stream**: Side-by-side or tabbed comparison of Supporting Evidence (Defender) vs Contradicting Evidence (Prosecutor).
   - **Quality Firewall Audit Drawer**: Inspecting source credibility, freshness decay, and pollution decontamination per citation.
