SELECT * FROM keyword AS k, movie_keyword AS mk, movie_companies AS mc, company_name AS cn, company_type AS ct, title AS t, movie_link AS ml WHERE cn.country_code != '[pl]' AND (cn.name LIKE '20th Century Fox%' OR cn.name LIKE 'Twentieth Century Fox%') AND ct.kind != 'production companies' AND ct.kind IS NOT NULL AND k.keyword IN ('sequel', 'revenge', 'based-on-novel') AND mc.note IS NOT NULL AND t.production_year > 1950 AND ml.movie_id = t.id AND t.id = ml.movie_id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id;