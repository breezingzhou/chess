# %%
from typing import Any, Generic, Optional, TypeVar, Type
from utils.common import WORKSPACE
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql.schema import Table
from contextlib import contextmanager


DB_PATH = WORKSPACE / "res/chess.sqlite"
MASTER_RES_PATH = WORKSPACE / "res/大师对局.csv"
DATABASE_URL = f"sqlite:///{DB_PATH}"


# echo can be set True for debug logging
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)
# expire_on_commit 设置为False，防止提交后对象被过期
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

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

  def _gen_order_by(self, order_by: list[Any]) -> list[Any]:
    order_exprs = []
    for ob in order_by:
      if isinstance(ob, str):
        desc = False
        name = ob
        if ob.startswith("-"):
          desc = True
          name = ob[1:]
        if not hasattr(self.model, name):
          continue
        col = getattr(self.model, name)
        order_exprs.append(col.desc() if desc else col.asc())
      else:
        # assume SQLAlchemy expression
        order_exprs.append(ob)
    return order_exprs

  def query(
      self,
      filters: Optional[list] = None,
      order_by: Optional[list] = None,
      limit: Optional[int] = None,
      offset: Optional[int] = None,
  ) -> list[ModelType]:
    """Generic query helper.

    Parameters
    - where: simple equality filters as a dict mapping column name -> value
    - filters: a list of SQLAlchemy expressions, e.g. [Model.col > 3]
    - order_by: list of column names (str) or SQLAlchemy column expressions. Use prefix '-' on column name for DESC.
    - limit: max number of rows to return
    - offset: number of rows to skip

    Returns a list of model instances.
    """
    results: list[ModelType] = []
    with get_session() as session:
      q = session.query(self.model)
      if filters:
        # filters expected to be an iterable of SQLAlchemy binary expressions
        q = q.filter(*filters)

      # order_by handling: accept strings or column expressions
      if order_by:
        order_exprs = self._gen_order_by(order_by)
        if order_exprs:
          q = q.order_by(*order_exprs)
      else:
        # default ordering by id asc
        if hasattr(self.model, "id"):
          q = q.order_by(self.model.id.asc())  # type:ignore

      if offset:
        q = q.offset(offset)
      if limit:
        q = q.limit(limit)

      results = q.all()

    return results

  def delete_by_ids(self, ids: list[int]) -> int:
    with get_session() as session:
      q = session.query(self.model).filter(self.model.id.in_(ids))  # type:ignore
      count = q.delete()
    return count

  def create_table(self) -> None:
    # create via metadata
    with get_session() as session:
      self.model.__table__.create(bind=session.bind, checkfirst=True)  # type:ignore

  def drop_table(self) -> None:
    with get_session() as session:
      self.model.__table__.drop(bind=session.bind, checkfirst=True)  # type:ignore

# %%
