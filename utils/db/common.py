# %%
from typing import Generic, Optional, TypeVar, Type
from utils.common import WORKSPACE
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql.schema import Table
from contextlib import contextmanager


DB_PATH = WORKSPACE / "res/chess.sqlite"
DATABASE_URL = f"sqlite:///{DB_PATH}"


# echo can be set True for debug logging
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)
# expire_on_commit 设置为False，防止提交后对象被过期
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

# # TODO不知道对不对
# BaseModelType = type(
#     "BaseModelType",
#     (object,),
#     {"__table__": Table}
# )
BaseModel = declarative_base()


@contextmanager
def get_session():
  session = SessionLocal()
  try:
    yield session
    session.commit()
  except Exception:
    session.rollback()
    raise
  finally:
    session.close()


ModelType = TypeVar("ModelType")


class BaseDAL(Generic[ModelType]):
  model: Type[ModelType]

  def __init__(self, model: Type[ModelType]):
    self.model = model

  def query_by_id(self, id_: int) -> Optional[ModelType]:
    with get_session() as session:
      model = session.get(self.model, id_)
    return model

  def create_table(self) -> None:
    # create via metadata
    with get_session() as session:
      self.model.__table__.create(bind=session.bind)  # type:ignore

  def drop_table(self) -> None:
    with get_session() as session:
      self.model.__table__.drop(bind=session.bind, checkfirst=True)  # type:ignore

# %%
