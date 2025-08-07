# %%


class Config:
  _instance = None

  dirichlet: float = 0.03
  epsilon: float = 0.25
  n_playout: int = 2000
  c_puct: float = 5
  temperature: float = 1e-3

  def __new__(cls):
    """单例模式实现"""
    if cls._instance is None:
      cls._instance = super(Config, cls).__new__(cls)
    return cls._instance


# %%
CONFIG = Config()

# %%
