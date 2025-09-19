from typing import Generic, Optional, TypeVar
from utils.common import WORKSPACE
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import contextmanager


DB_PATH = WORKSPACE / "res/chess.sqlite"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# echo can be set True for debug logging
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
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
  # model: BaseModel

  def __init__(self, model: ModelType):
    self.model = model

  def query_by_id(self, id_: int) -> Optional[ModelType]:
    with get_session() as session:
      return session.get(self.model, id_)

  def create_table(self) -> None:
    # create via metadata
    with get_session() as session:
      self.model.__table__.create(bind=session.bind)

  def drop_table(self) -> None:
    with get_session() as session:
      self.model.__table__.drop(bind=session.bind, checkfirst=True)
