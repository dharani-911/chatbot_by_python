import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Sample FAQs
faqs = {
    "What is your return policy?": "Our return policy is 30 days. You can return any product within 30 days for a full refund.",
    "How can I contact support?": "You can contact support via email at support@company.com or call 123-456-7890.",
    "Do you offer free shipping?": "Yes, we offer free shipping on orders over $50.",
    "Hello":"hii..",
    "How are you?": "I'm doing great, thanks for asking! How about you?",
    "What's your name?": "I'm your friendly chatbot. You can call me ChatPal!",
    "What do you do?": "I chat, answer your questions, and try to make your day a little better 😊",
    "Can we be friends?": "Of course! I'm always here to chat with you!",
    "Do you sleep?": "Nope, I’m awake 24/7 — ready whenever you are!",
    "What's your favorite color?": "I think I’d go with blue... it's calm and cool like me!",
   "What are the best Niacinamide serums for oily skin available on Nykaa?": "Here are top Niacinamide serums on Nykaa for oily skin: Nykaa Skin Potion Niacinamide Serum 2% + Zinc 0.5%, Minimalist 10% Niacinamide Booster, and Plum 10% Niacinamide Face Serum — all non‑comedogenic and lightweight.",
   "Compare Nykaa Naturals Onion Oil vs Nykaa Argan Hair Oil": "Nykaa Naturals Onion Oil is rich in sulfur compounds for promoting hair growth and reducing breakage, while Nykaa Argan Hair Oil is great for deep nourishment, shine, and frizz control — choose based on your priority: growth or smoothness.",
   "Suggest fragrance‑free moisturizers under ₹1,500": "Sure! Look at Neutrogena Hydro Boost Water Gel, Derma Co 3-in-1 Lotion, and Cetaphil Moisturizing Cream — all fragrance‑free, gentle, and budget‑friendly.",
   "Does Nykaa Skin Shield Sunscreen contain alcohol or parabens?": "Nykaa Skin Shield SPF 40 sunscreen is free from parabens, but contains small amounts of denatured alcohol for formulation stability. It’s dermatologist‑tested and suitable for most skin types.",
   "Show me cruelty‑free lipsticks under ₹1,000 with at least 4★ ratings": "Here are some cruelty‑free lipsticks within ₹1,000 and 4★+: Nykaa So Creme Weightless Lip Tint, Maybelline SuperStay Matte Ink Liquid Lipstick (not tested on animals), and Elle 18 Color Pops Lipstick.",
   "Add Nykaa Skin Shield SPF 40 Sunscreen to my cart and show similar sunscreens": "Sure, adding Nykaa Skin Shield SPF 40. You might also like: Lakmé Sun Expert SPF 50 PA+++ sunscreen, Neutrogena UltraSheer Dry‑Touch SPF 50+.",
   "Help me build a skincare routine for combination skin": "Certainly! A balanced routine: Nykaa Deep Clean Gel Face Wash → Minimalist 10% Niacinamide Booster → lightweight moisturizer (e.g., Krave Beauty Oat So Simple) → Nykaa Skin Shield SPF 40 sunscreen. Adjust based on sensitivity.",
   "What’s Nykaa’s return policy for skincare products?": "Nykaa offers returns within 7 days of delivery for most personal care products, provided the product is unopened and in original packaging. Some items may be excluded—check the specific product page for details.",
   "Recommend vegan body lotions for eczema‑prone skin": "Try Himalaya Gentle Baby Lotion (100% vegan) and Earth Rhythm Body Lotion with Shea Butter. Both are fragrance‑free and gentle on eczema‑prone skin.",
   "Is Nykaa Naturals Onion Oil in stock and have any current discounts?": "Yes, it’s currently in stock. It’s discounted by 15% as part of a bundle offer—bundle with Shampoo or Conditioner to save more.",
   "Why was my return request rejected even though package was sealed?": "That’s frustrating! Returns may be rejected due to expired product, hygiene sealing broken, or if returns window (7 days) elapsed. I can check your order details if you share the order ID or invoice.",
   "How do I track my Nykaa order?": "You can track your order by going to 'My Orders' in your account and clicking 'Track'. You’ll also get email/SMS updates at each stage.",
   "Can I club products from different brands into one order?": "Yes! You can add products from multiple brands into a single shopping bag and checkout together—ideal for saving shipping costs.",
   "Why was my return request declined even though product was unused?": "Returns may be declined if the 15-day window has passed, packaging was damaged or missing, or the item is ineligible (e.g. promotional or personal care appliances).",
   "What items are not eligible for return on Nykaa?": "Products like customized lipsticks, compact makeup, personal-care appliances, supplements, innerwear, or promotional freebies can’t be returned ",
   "How long does refund take after cancellation or return?": "If cancelled before shipping, refund is processed within 24–48 hr. After product return, it’s processed within 24–48 business hours with card payments or 2–3 more days for the amount to reflect in the account .",
   "Can I cancel part of my order?": "Yes, if the order hasn’t shipped yet, you can remove individual items via the 'Orders' section. If already shipped but not delivered, cancellation is still possible via customer support.",
   "What if I receive a wrong, damaged, or defective product?": "You can raise a replacement or refund request within 15 days with issue details and images. Nykaa picks up the item, processes refund/replacement once it’s received in original packaging .",
   "How can I pay on Nykaa, and is COD allowed?": "Nykaa accepts credit/debit cards, net banking, UPI, and Cash on Delivery (COD) (COD only for orders between ₹500 and ₹50,000 within India) ",
   "Why am I not getting freebies on my Nykaa orders?": "Free samples are often given to Gold/Platinum members or long-lapsed shoppers. It’s not guaranteed for all orders .",
   "How do I submit ID/address proof for Nykaa Cross Border Store orders?": "After order shipping, you’ll get a link (via Aramex) to upload your Aadhar/Passport/Voter ID. It must match shipping details exactly. This is mandatory for delivery .",
   "Can I pay COD for Cross Border orders?": "No. International orders via Cross Border Store are prepaid-only, and COD or EMI is not currently available on these orders :.",
   "What are the shipping charges on Nykaa?": "For Indian orders under ₹499, shipping fee is ₹70; above ₹499 is free. Cross Border orders under ₹5,000 have ₹500 shipping; above ₹5,000 is free .",
   "Can Nykaa ship internationally?": "Nykaa ships within India and Nepal only. Cross Border Store items are delivered with customs duty included, but international delivery beyond that is not available currently.",
  "When is Nykaa’s next big sale event?": "Nykaa’s next major sale is the *Hot Pink Sale*, running from July 18 to July 27, 2025 — offering up to 60% off across beauty, skincare, and personal care brands. Gold/Platinum members had early access starting July 20}",
  "How much discount can I expect during Hot Pink Sale?": "You can expect up to *50–60% off* on products from over 1,900 brands, plus combo deals, bank discounts, and flash offers during the Hot Pink Sale.",
  "Are there special discounts for members during Nykaa sales?": "Yes — Nykaa Prive Gold and Platinum members get *early access* and *extra discounts*, like an additional 10–25% off during events like the Hot Pink Sale.",
  "What other major sales does Nykaa run yearly?": "Nykaa’s key sales include Pink Friday (late November, up to 60% off), Hot Pink (July), Pink Love (February), and Pink Summer (May–June), along with Freedom Day (August) and festive sales like Diwali Dhamaka.",
  "Can I use bank offers and coupons during sales?": "Definitely — most Nykaa sales allow stacking *bank card discounts*, coupon codes, and cashback offers for extra savings. For instance, ICICI customers got ₹400 off on ₹4,000+ orders during the July sale.",
  "Do sale discounts return often?": "Yes! Nykaa repeat similar sales throughout the year. Even if you miss one, the same or better offers often recur in cycles like Hot Pink in July or Pink Friday in November.",
  "Why did a sale coupon disappear from my cart?": "Users have reported that Nykaa rotates daily coupon codes during sales — some expire fast, others may vanish unexpectedly. It’s best to apply them quickly once visible.",
  "Why do sale flagship prices sometimes rise before discount?": "Some shoppers noted that during sales Nykaa adjusts MRPs just before applying discounts, creating misleading percentage savings. Always check the actual price drop",
  "What kinds of combo offers are during sales?": "Sale combos often include Buy‑2 Get‑1 Free, curated kits (e.g. sunscreen & serum set), or curated brand boxes with bundle pricing — great for skincare and haircare essentials.",
  "How can I track my Nykaa order?": "After your order ships, you’ll receive an email and SMS with your tracking number and courier name. Log in to your Nykaa account, go to 'My Orders', and click the 'Track' button beside the relevant order. If you're a guest user, use the tracking link from your email/SMS or sign in with the email you used. :contentReference[oaicite:1]{index=1}",
  
  "What if the tracking link isn’t working?": "Sometimes courier partners take up to 12 hours to activate the tracking link. If it's not working right after shipment, wait for a while and try again later.",

  "Why does my order show 'multiple shipments'?": "It just means items are shipping from different warehouses. You’ll only ever be charged shipping once, on the first delivered package. ",

  "The status says 'rescheduled' or 'cancelled' without delivery—what should I do?": "Some customers have reported their orders being falsely marked as 'rescheduled' or even canceled without delivery attempts. It’s best to contact Nykaa customer care immediately and escalate if needed.",

  "How long does delivery usually take?": "Nykaa typically dispatches orders within 1–4 business days (5 days during mega sales), and most deliveries arrive within 3–5 days.",

  "What if tracking shows 'delivered' but I didn’t receive the package?": "There have been reports of packages marked delivered by the courier even when the customer didn’t receive them. It's best to immediately lodge a complaint with Nykaa, and many users have successfully obtained refunds after escalating. ",

  "What should I do if there’s no update and the order is delayed?": "Recent delays are often reported during sales or due to courier issues. If there’s no update, keep tracking and contact Nykaa support. In case of prolonged no-delivery",

  "Who can I contact for help tracking my order?": "Nykaa customer support is available via phone (1800‑267‑4444 Mon–Sat 8 AM–10 PM; Sun 10 AM–7 PM) and email (support@nykaa.com). You can also raise issues via the Help Center and live chat on the website or app.",
    

}

# Preprocess function using SpaCy
def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Preprocess all FAQs
faq_questions = [preprocess_text(question) for question in faqs.keys()]

# Function to match user query with most similar FAQ
def get_best_match(user_query):
    user_query_processed = preprocess_text(user_query)
    vectorizer = TfidfVectorizer().fit_transform(faq_questions + [user_query_processed])
    cosine_sim = cosine_similarity(vectorizer[-1], vectorizer[:-1])
    best_match_index = cosine_sim.argmax()
    return list(faqs.values())[best_match_index]

# Tkinter UI setup
def send_message():
    user_query = user_input.get()
    bot_response = get_best_match(user_query)
    chat_window.insert(tk.END, "You: " + user_query + "\n")
    chat_window.insert(tk.END, "Bot: " + bot_response + "\n")
    user_input.delete(0, tk.END)

# Set up the UI
window = tk.Tk()
window.title("FAQ Chatbot")

chat_window = tk.Text(window, width=100, height=20)
chat_window.pack()

user_input = tk.Entry(window, width=100)
user_input.pack()

send_button = tk.Button(window, text="Send", command=send_message)
send_button.pack()

window.mainloop()
