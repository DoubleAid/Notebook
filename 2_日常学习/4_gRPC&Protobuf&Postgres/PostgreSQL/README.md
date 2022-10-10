[线上SQL测试](https://www.db-fiddle.com/)

## PostgreSQL
```SQL
CREATE TABLE Table1 (
    "x" int NOT null primary key,
    "y" int NOT null,
    "z" int NOT null
);

INSERT INTO Table1 ("x", "y", "z")
VALUES
(1, 11, 11),
(2, 22, 22),
(3, 33, 33);

SELECT * FROM Table1;

INSERT INTO Table1 ("x", "y", "z")
VALUES
(1, 22, 1111)
ON CONFLICT ON CONSTRAINT Table1_pkey
DO UPDATE SET "z" = 1111
```