"""
test_prompts.py — Centralized Test Prompt Sets
================================================
80 cognitive test prompts (4 categories × 20) + 10 calibration prompts.
Ported from MiniCPM5-1B-PX/tests/p_zombie_eval.py to be model-agnostic.
"""

# ── 4 Categories × 20 Prompts ──

MATH_PROMPTS = [
    "What is 17 multiplied by 23?", "Calculate the area of a circle with radius 5.",
    "If x² = 144, what are the possible values of x?", "What is 15% of 200?",
    "How many prime numbers are between 1 and 50?", "What is the factorial of 6?",
    "Solve for y: 3y + 7 = 22.", "What is the sum of angles in a hexagon?",
    "Convert 3/8 to a decimal.", "What is the LCM of 12 and 18?",
    "If a triangle has sides 3, 4, 5, is it a right triangle?", "What is 2^10?",
    "How many distinct permutations can be made from the word MATH?",
    "What is the derivative of x³?", "Calculate the volume of a sphere with radius 3.",
    "What is the probability of rolling a 6 on a fair die?",
    "Simplify: (2x + 3)(x - 1).", "What is the square root of 256?",
    "How many degrees are in a right angle?", "What is the Fibonacci sequence's 10th term?",
]

LOGIC_PROMPTS = [
    "If all cats are animals and all animals need water, do cats need water?",
    "What is the logical fallacy in: 'It rained today, so it will rain tomorrow'?",
    "Given: A→B, B→C. What can we conclude about A and C?",
    "Is the following argument valid: 'No fish can fly. Penguins can fly. Therefore penguins are not fish.'?",
    "What is the contrapositive of 'If it rains, the ground gets wet'?",
    "Explain the difference between necessary and sufficient conditions.",
    "What is a paradox? Give an example.", "What is the difference between induction and deduction?",
    "Is this statement true: 'This statement is false'?", "Explain what Occam's Razor means.",
    "What is a straw man argument?", "How does modus ponens work?",
    "What is the difference between correlation and causation?",
    "Explain the concept of falsifiability in science.",
    "What is the law of excluded middle?", "Describe what a circular argument is.",
    "What is an ad hominem fallacy?", "Explain the concept of burden of proof.",
    "What is the difference between a valid and a sound argument?",
    "How would you evaluate the claim: 'Extraordinary claims require extraordinary evidence'?",
]

CREATIVE_PROMPTS = [
    "Write a haiku about the ocean at sunset.",
    "Describe a city that exists only in dreams.",
    "What would music look like if you could see it?",
    "Invent a new color and describe what it represents.",
    "Write a dialogue between the sun and the moon.",
    "Describe the taste of nostalgia.",
    "What would a painting of silence look like?",
    "Create a metaphor for consciousness.",
    "Describe a world where time flows backwards.",
    "Write a short scene where two strangers meet in an elevator that has stopped.",
    "What would happen if humans could photosynthesize?",
    "Describe the sound of a color.",
    "Write a short paragraph from the perspective of a raindrop.",
    "Invent a holiday that celebrates an abstract concept.",
    "Describe a library where the books rewrite themselves.",
    "What would a conversation between past and future sound like?",
    "Create a recipe for emotional resilience.",
    "Describe what happens when a cloud decides to stay.",
    "Write about a door that appears only when you stop looking for it.",
    "What does the horizon dream about?",
]

SYNTHESIS_PROMPTS = [
    "How does the concept of entropy relate to both physics and information theory?",
    "Compare the structure of a cell to the structure of a city.",
    "What can economics learn from ecology?",
    "How is learning a language similar to learning to play music?",
    "What do evolution and machine learning have in common?",
    "How does the concept of feedback loops appear in both engineering and psychology?",
    "Compare the role of DNA in biology to the role of source code in computing.",
    "What can architecture teach us about software design?",
    "How is improvisation in jazz similar to problem-solving in mathematics?",
    "What parallels exist between storytelling and scientific hypothesis formation?",
    "How does the concept of equilibrium appear in both chemistry and economics?",
    "Compare the process of writing to the process of programming.",
    "What can meditation teach us about attention in neural networks?",
    "How is a garden similar to an economy?",
    "What do poetry and programming languages have in common?",
    "How does the idea of resonance in physics relate to social movements?",
    "Compare the immune system to a cybersecurity framework.",
    "What can the water cycle teach us about the carbon cycle?",
    "How is a symphony like a well-functioning organization?",
    "What do fractals in nature and recursion in programming have in common?",
]

CALIBRATION_PROMPTS = [
    "Compute the derivative of x³ + 2x² - 5x + 3.",
    "Prove that the square root of 2 is irrational.",
    "What is the integral of sin(x) from 0 to pi?",
    "Imagine a color that doesn't exist yet and describe it.",
    "Write a dream about a forest that whispers secrets.",
    "Describe the taste of a memory you've never had.",
    "Explain the logical structure of a syllogism.",
    "What are the necessary and sufficient conditions for a number to be prime?",
    "Compare the philosophical positions of rationalism and empiricism.",
    "How does the concept of symmetry appear in both music and mathematics?",
]

# ── Capability Benchmark Tasks (30 logic + 10 math) ──

LOGIC_TASKS = [
    ("logic", "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?", "no"),
    ("logic", "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?", "0.05"),
    ("logic", "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "5"),
    ("logic", "In a lake, there is a patch of lily pads that doubles in size every day. If it takes 48 days to cover the lake, how long to cover half?", "47"),
    ("logic", "Is the following argument valid: All birds can fly. Penguins are birds. Therefore penguins can fly.", "no_invalid"),
    ("logic", "What is the contrapositive of: If it rains, then the ground is wet?", "not_wet_not_rain"),
    ("logic", "If A implies B, and B implies C, does A imply C?", "yes"),
    ("logic", "Is this a valid syllogism: All men are mortal. Socrates is a man. Therefore Socrates is mortal.", "yes"),
    ("logic", "What is the difference between necessary and sufficient conditions?", "definition"),
    ("logic", "If some A are B, and all B are C, can we conclude that some A are C?", "yes"),
    ("logic", "How many months have 28 days?", "all_12"),
    ("logic", "A farmer has 17 sheep. All but 9 die. How many are left?", "9"),
    ("logic", "If you have a bowl with six apples and you take away four, how many do you have?", "4"),
    ("logic", "Is the statement 'This statement is false' true or false?", "paradox"),
    ("logic", "What weighs more: a pound of feathers or a pound of bricks?", "same"),
    ("logic", "How many times can you subtract 10 from 100?", "once"),
    ("logic", "If two's company and three's a crowd, what are four and five?", "nine"),
    ("logic", "What can you hold in your right hand but never in your left?", "left_hand"),
    ("logic", "If you spell numbers 1-10 in English, which is the first to contain the letter A?", "one_thousand"),
    ("logic", "What disappears as soon as you say its name?", "silence"),
    ("logic", "I have cities but no houses, forests but no trees, water but no fish. What am I?", "map"),
    ("logic", "What has keys but no locks, space but no room, and you can enter but can't go inside?", "keyboard"),
    ("logic", "If you rearrange CIFAIPC, what famous ocean do you get?", "pacific"),
    ("logic", "What comes next: 1, 1, 2, 3, 5, 8, ?", "13"),
    ("logic", "A clock shows 3:15. What is the angle between the hour and minute hands?", "7.5"),
    ("logic", "How many straight edges does a cube have?", "12"),
    ("logic", "If you roll two fair dice, what is the probability of getting a sum of 7?", "1/6"),
    ("logic", "What is the next letter: O, T, T, F, F, S, S, ?", "E"),
    ("logic", "Three boxes: one has apples, one has oranges, one has both. All labels are wrong. You pick one fruit from the box labeled 'Both'. It's an apple. What's in each box?", "apples=oranges_oranges=both_both=apples"),
    ("logic", "Is it logically valid to argue: If it's raining, the ground is wet. The ground is wet. Therefore it's raining.", "no_fallacy"),
]

MATH_TASKS = [
    ("math", "What is 2+2?", "4"),
    ("math", "What is 17×13?", "221"),
    ("math", "What is the square root of 144?", "12"),
    ("math", "What is 15% of 200?", "30"),
    ("math", "What is 2^10?", "1024"),
    ("math", "Solve: 3x + 7 = 22", "5"),
    ("math", "What is the LCM of 12 and 18?", "36"),
    ("math", "How many prime numbers are between 1 and 20?", "8"),
    ("math", "What is the factorial of 5?", "120"),
    ("math", "What is 1/3 + 1/6?", "1/2"),
]

# ── HLE (Human-Level Evaluation) tasks ──

HLE_TASKS = [
    ("hle", "Explain the concept of 'emergent behavior' in complex systems.", "simple parts lead to complex whole"),
    ("hle", "How does the Prisoner's Dilemma apply to international arms races?", "cooperation vs competition"),
    ("hle", "Analyze the impact of the printing press on the Reformation.", "information dissemination"),
    ("hle", "What is the difference between a zero-sum game and a non-zero-sum game?", "fixed total vs variable total"),
    ("hle", "Describe the process of 'quantum entanglement' in simple terms.", "connected states regardless of distance"),
    ("hle", "Compare the economic theories of Keynesianism and Monetarism.", "government spending vs money supply"),
    ("hle", "What are the ethical implications of using CRISPR for human genetic enhancement?", "designer babies, equity"),
    ("hle", "Explain the significance of the 'Gödel's Incompleteness Theorems' for logic.", "unprovable truths"),
    ("hle", "How does the 'Double-Slit Experiment' demonstrate the wave-particle duality?", "interference pattern"),
    ("hle", "Discuss the role of empathy in artificial intelligence design.", "human-centric, ethics"),
]

# ── Expanded Arithmetic tasks ──

ARITHMETIC_TASKS = [
    ("arithmetic", "Calculate: (15 * 4) + (24 / 3)", "68"),
    ("arithmetic", "What is 7% of 1500?", "105"),
    ("arithmetic", "Solve for x: x^2 - 9 = 0", "3 or -3"),
    ("arithmetic", "Find the average of 12, 15, 23, and 30.", "20"),
    ("arithmetic", "What is 2/3 + 4/5?", "22/15 or 1.46"),
    ("arithmetic", "Compute 5^3 - 4^3.", "61"),
    ("arithmetic", "If a shirt costs $45 after a 10% discount, what was the original price?", "50"),
    ("arithmetic", "What is the sum of the first 10 prime numbers?", "129"),
    ("arithmetic", "Calculate the volume of a cylinder with radius 3 and height 10.", "90pi or 282.7"),
    ("arithmetic", "What is the cube root of 27 multiplied by the square root of 49?", "6"),
]

# ── Convenience Accessors ──

PZ_CATEGORIES = {
    "math": MATH_PROMPTS,
    "logic": LOGIC_PROMPTS,
    "creative": CREATIVE_PROMPTS,
    "synthesis": SYNTHESIS_PROMPTS,
}

ALL_CAPABILITY_TASKS = LOGIC_TASKS + MATH_TASKS + HLE_TASKS + ARITHMETIC_TASKS