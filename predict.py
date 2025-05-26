print("🔍 Starting predict.py...")

try:
    import groundingdino
    from groundingdino.util.inference import load_model as load_dino_model, predict as dino_predict
    print("✅ groundingdino import succeeded")
except Exception as e:
    print(f"❌ groundingdino import FAILED: {e}")
    raise e


def predict() -> dict:
    print("🚀 predict() function reached")
    return {"status": "ok"}
