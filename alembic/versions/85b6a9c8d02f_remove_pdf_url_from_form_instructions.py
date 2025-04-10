"""remove_pdf_url_from_form_instructions

Revision ID: 85b6a9c8d02f
Revises: 63ffaa962d7a
Create Date: 2025-04-10 09:55:20.585219

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '85b6a9c8d02f'
down_revision: Union[str, None] = '63ffaa962d7a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema using batch mode for SQLite."""
    # Use batch_alter_table for SQLite compatibility
    with op.batch_alter_table('form_fields', schema=None) as batch_op:
        # Re-establish foreign key with CASCADE
        batch_op.create_foreign_key(
            'fk_form_fields_instruction_id', 'form_instructions', ['instruction_id'], ['id'], 
            ondelete='CASCADE'
        )
    
    # Drop the pdf_url column
    with op.batch_alter_table('form_instructions', schema=None) as batch_op:
        batch_op.drop_column('pdf_url')


def downgrade() -> None:
    """Downgrade schema using batch mode for SQLite."""
    # Add the pdf_url column back
    with op.batch_alter_table('form_instructions', schema=None) as batch_op:
        batch_op.add_column(sa.Column('pdf_url', sa.VARCHAR(length=255), nullable=True))
    
    # Restore the original foreign key without CASCADE
    with op.batch_alter_table('form_fields', schema=None) as batch_op:
        batch_op.create_foreign_key(
            'fk_form_fields_instruction_id', 'form_instructions', ['instruction_id'], ['id']
        )
