# Farcaster Trending Topics: Final Approach Summary

## Overview

We developed a system to detect trending topics on Farcaster social media platform through incremental enhancements to a direct LLM analysis approach. This document summarizes our final methodology and key findings.

## Key Innovations

1. **Batch Processing Architecture**
   - Process entire dataset in batches of 15,000 posts
   - Each batch analyzed separately to identify 10 potential topics
   - Results consolidated via a second LLM pass

2. **Top-level Post Filtering**
   - Focus only on original posts, not replies
   - Significantly improved topic quality by focusing on content with context
   - Average text length improved from ~45 to ~93 characters

3. **Engagement Scoring with Conversation Metrics**
   - Enhanced weighting that prioritizes posts with rich conversations
   - Combined formula: `likes + (3 × recasts) + (5 × reply_count) + (10 × unique_repliers)`
   - Effectively surfaced community-focused content like "Eggman Virus" and "Base is for..."

4. **Prompt Engineering for Discovery**
   - Emphasis on novelty and surprise value
   - Specific guidance to avoid token/crypto focused content
   - Clear instructions to prioritize platform-wide cultural behaviors

5. **Model Selection for Final Consolidation**
   - Two viable options with different tradeoffs:
     - **Gemini 2.5 Pro**: Superior topic diversity and quality (89s processing time)
     - **Gemini 1.5 Pro**: Reasonable quality with 3.2× faster processing (27s)

## Optimal Approach

Our testing revealed the following optimal configuration:

1. **Data Processing**:
   - Process the entire dataset in batches of 15,000 posts
   - Apply top-level post filtering to each batch
   - Use enhanced engagement scoring with conversation metrics

2. **Batch LLM Analysis**:
   - Model: Gemini 2.0 Flash Lite (good balance of speed and quality)
   - Request 10 topics per batch for comprehensive coverage
   - Save all batch results for final consolidation

3. **Topic Consolidation**:
   - **For highest quality**: Gemini 2.5 Pro 
   - **For speed-sensitive applications**: Gemini 1.5 Pro
   - Prompt focused on novelty, broad relevance, and avoiding promotional content

## Results Quality Comparison

The key differentiating factor between our approaches was topic diversity and "clickability":

### Early Baseline:
```
Topic 1: Farcaster Activity (15%)
Topic 2: Token/Crypto Discussion (20%)
Topic 3: Welcome Back/Greetings (10%)
Topic 4: Personal Life/Interactions (15%)
Topic 5: Project/Product Promotion (10%)
```

### Final Approach with Gemini 2.5 Pro:
```
Topic 1: Farcaster Tipping Culture Flourishes (10%)
Topic 2: ITAP: Sharing Our World Daily (8%)
Topic 3: Daily Mystery Location Challenge (8%)
Topic 4: Defining "Base is for..." (5%)
Topic 5: Users Ghibli-fy Profile Pictures (4%)
```

### Final Approach with Gemini 1.5 Pro:
```
Topic 1: Farcaster Tipping Culture (12%)
Topic 2: Introducing Tipping Feature (15%)
Topic 3: Hiring Farcaster Devs (10%)
Topic 4: ITAP (I took a photo) (12%)
Topic 5: Monochrome Monday (6%)
```

## Conclusion

The batch processing approach with Gemini 2.5 Pro for consolidation produces the highest quality results, identifying specific, diverse, and genuinely interesting topics that would be compelling for users to explore. The approach successfully:

1. Avoids token/crypto promotional content
2. Focuses on platform-wide behavioral trends
3. Highlights creative expression and cultural phenomena
4. Balances content types across multiple aspects of the platform

For production systems where processing time is critical, Gemini 1.5 Pro offers a reasonable alternative with a 3.2× speed advantage, though with slightly less diverse topic selection.

The final system successfully improves upon all the limitations of our baseline approach, moving from generic categories to specific, engaging topics that reveal genuine platform activity.