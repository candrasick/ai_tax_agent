"""add_index_to_form_fields_instruction_id

Revision ID: 63ffaa962d7a
Revises: 59b5151e47d5
Create Date: 2025-04-10 09:50:20.545724

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector


# revision identifiers, used by Alembic.
revision: str = '63ffaa962d7a'
down_revision: Union[str, None] = '59b5151e47d5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema using batch mode for SQLite."""
    # Get connection and inspector
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    
    # Check if index already exists (it might due to previous migration attempt)
    table_indexes = inspector.get_indexes('form_fields')
    index_exists = any(idx['name'] == 'ix_form_fields_instruction_id' for idx in table_indexes)
    
    # First batch operation for index (if needed)
    if not index_exists:
        with op.batch_alter_table('form_fields', schema=None) as batch_op:
            batch_op.create_index('ix_form_fields_instruction_id', ['instruction_id'], unique=False)
    
    # Second batch operation for foreign key
    with op.batch_alter_table('form_fields', schema=None) as batch_op:
        # SQLite batch mode will handle dropping and recreating the constraint
        batch_op.create_foreign_key(
            None, 'form_instructions', ['instruction_id'], ['id'], 
            ondelete='CASCADE'
        )


def downgrade() -> None:
    """Downgrade schema."""
    conn = op.get_bind()
    
    with op.batch_alter_table('form_fields', schema=None) as batch_op:
        # Recreate the default foreign key without CASCADE
        batch_op.create_foreign_key(
            None, 'form_instructions', ['instruction_id'], ['id']
        )
        # Try to drop the index (may not exist if upgrade was partial)
        try:
            batch_op.drop_index('ix_form_fields_instruction_id')
        except:
            pass
