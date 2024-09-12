-- List of access methods

CREATE ACCESS METHOD rabbithole TYPE INDEX HANDLER _rabbithole_amhandler;
COMMENT ON ACCESS METHOD rabbithole IS 'rabbithole index access method';

-- List of operator families

CREATE OPERATOR FAMILY vector_l2_ops USING rabbithole;

-- List of operator classes

CREATE OPERATOR CLASS vector_l2_ops
    FOR TYPE vector USING rabbithole FAMILY vector_l2_ops AS
    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (vector, sphere_vector) FOR SEARCH;
