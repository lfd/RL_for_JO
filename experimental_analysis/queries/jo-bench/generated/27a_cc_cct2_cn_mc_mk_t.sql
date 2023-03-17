SELECT * FROM comp_cast_type AS cct2, complete_cast AS cc, title AS t, movie_keyword AS mk, movie_companies AS mc, company_name AS cn WHERE cct2.kind = 'complete' AND cn.country_code != '[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND mc.note IS NULL AND t.production_year BETWEEN 1950 AND 2000 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_id = cn.id AND cn.id = mc.company_id AND t.id = cc.movie_id AND cc.movie_id = t.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id;